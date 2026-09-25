"""Frozen vision-tower features, computed once per image and reused by every epoch, round and eval.

LoRA (``sf2.lora``) trains only the SmolVLM text layers, so the vision tower + connector map a frame to the same
``[image_seq_len, d]`` features forever. Recomputing them was 58-61% of a training step on MPS; feeding cached
features through ``VLMDecisionModel.forward(image_hidden_states=...)`` gives identical logits (fp16 storage moves
them by < 1e-3) at 2.4x the step rate.

Layout: ``<data dir>/vision/<fingerprint>.npy`` (fp16 ``[n_images, seq, d]``, memory-mapped) and
``<fingerprint>.json`` (image path -> row; written last, so its presence means the cache is complete). The
fingerprint hashes the vision + connector weights and the image preprocessing, so a checkpoint that changed either
simply has no cache yet.

``install`` swaps laya's training/eval item builders for the cached ones (and, with or without a cache, makes the
training stream draw each example once per pass instead of with replacement).
"""
import functools
import hashlib
import json
import os
import random
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

import laya.vlm_train as vt
from laya.preprocess import prefix_ids
from laya.vlm import MASK_PREFIX_TEXT, PREFIX_TEXT, _load_image, build_vlm_inputs, collate_vlm, processor_readout

_ORIG = {"forward": vt._forward, "ItemStream": vt.ItemStream}


def image_key(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def example_images(ex: Dict) -> List[str]:
    st = ex["state"] if isinstance(ex.get("state"), dict) else {}
    return ([st["image"]] if st.get("image") else []) + list(st.get("images") or [])


def fingerprint(model, prep) -> str:
    """Hash of everything that decides an image's features: vision + connector weights and preprocessing."""
    h = hashlib.sha256()
    enc = model.encoder
    for name, mod in (("vision", enc.vision_model), ("connector", enc.connector)):
        for pname, p in sorted(mod.state_dict().items()):
            h.update(("%s.%s:%s:%s" % (name, pname, tuple(p.shape), p.dtype)).encode())
            h.update(p.detach().to("cpu").contiguous().numpy().tobytes())
    h.update(repr((prep.image_size, prep.backend, prep.interpolation, prep.split_edge)).encode())
    return h.hexdigest()[:16]


def _paths(data_dir: str, fp: str):
    d = os.path.join(data_dir, "vision")
    return os.path.join(d, fp + ".npy"), os.path.join(d, fp + ".json")


def missing(data_dirs: Sequence[str], fp: str) -> List[str]:
    return [d for d in data_dirs if not os.path.exists(_paths(d, fp)[1])]


def build(data_dir: str, examples: Sequence[Dict], encode: Callable[[List[str]], np.ndarray], fp: str,
          batch: int = 4096, log: Callable[[str], None] = lambda s: None) -> None:
    """Encode every distinct image the examples use (once each) into ``data_dir``'s cache. No-op when complete."""
    npy, idx = _paths(data_dir, fp)
    if os.path.exists(idx):
        return
    keys = sorted({image_key(p) for ex in examples for p in example_images(ex)})
    os.makedirs(os.path.dirname(npy), exist_ok=True)
    tmp = npy + ".partial.npy"
    out = None
    for s in range(0, len(keys), batch):
        feats = np.asarray(encode(keys[s: s + batch]), dtype=np.float16)
        if out is None:
            out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(keys),) + feats.shape[1:])
        out[s: s + len(feats)] = feats
        log("vision cache %s: %d/%d images" % (os.path.basename(os.path.normpath(data_dir)), s + len(feats), len(keys)))
    if out is not None:
        out.flush()
        del out
        os.replace(tmp, npy)
    with open(idx + ".tmp", "w") as f:
        json.dump({"fingerprint": fp, "images": keys}, f)
    os.replace(idx + ".tmp", idx)


class FeatureCache:
    """Image path -> cached features. Pickles as its index only; each process maps the arrays on first use."""

    def __init__(self, files: List[str], index: Dict[str, tuple]):
        self.files, self.index, self._mm = files, index, None

    def __getstate__(self):
        return {"files": self.files, "index": self.index, "_mm": None}

    def get(self, paths: Sequence[str]) -> torch.Tensor:
        if self._mm is None:
            self._mm = [np.load(f, mmap_mode="r") for f in self.files]
        rows = [self.index[image_key(p)] for p in paths]
        return torch.from_numpy(np.stack([self._mm[fi][r] for fi, r in rows]))


def load(data_dirs: Sequence[str], fp: str) -> FeatureCache:
    files, index = [], {}
    for d in data_dirs:
        npy, idx = _paths(d, fp)
        with open(idx) as f:
            keys = json.load(f)["images"]
        if keys:
            files.append(npy)
            for r, k in enumerate(keys):
                index.setdefault(k, (len(files) - 1, r))
    return FeatureCache(files, index)


# --- building features with the model --------------------------------------------------------------------------


class _Pixels(torch.utils.data.Dataset):
    def __init__(self, processor, paths):
        self.processor, self.paths = processor, paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from laya.vlm import vlm_prefix

        pre = vlm_prefix(self.processor, [_load_image(self.paths[i])])
        if pre["raw_images"] is not None:
            return {"raw": pre["raw_images"][0]}
        return {"pv": pre["pixel_values"][0], "pam": pre["pixel_attention_mask"][0]}


def encoder(model, processor, batch_size: int = 32, num_workers: int = 4) -> Callable[[List[str]], np.ndarray]:
    """``paths -> fp16 [n, seq, d]`` with the model's own preprocessing; images are decoded in loader workers."""
    dev = next(model.parameters()).device

    @torch.no_grad()
    def encode(paths):
        loader = torch.utils.data.DataLoader(_Pixels(processor, paths), batch_size=batch_size,
                                             num_workers=num_workers, worker_init_fn=vt._single_thread_worker)
        out = []
        was_training = model.training
        model.eval()
        for b in loader:
            if "raw" in b:
                f = model.encode_raw_images(b["raw"].to(dev))
            else:
                f = model.encode_images(b["pv"].to(dev, model.encoder.dtype), b["pam"].to(dev))
            out.append(f.to(torch.float16).cpu().numpy())
        model.train(was_training)
        return np.concatenate(out)

    return encode


# --- items, batches, forward -----------------------------------------------------------------------------------


def cached_item(processor, ex: Dict, rng: random.Random, cache: FeatureCache, image_seq_len: int,
                shuffle: bool = True, order: Optional[List[int]] = None) -> Dict:
    """``laya.vlm_train.make_item`` without touching a pixel: same ids and targets, plus ``"feats"``."""
    k = len(vt.render_options(ex["q"]))
    if order is None:
        order = list(range(k))
        if shuffle:
            rng.shuffle(order)
    paths = example_images(ex)
    state = {kk: v for kk, v in ex["state"].items() if kk not in ("image", "images")} \
        if isinstance(ex["state"], dict) else ex["state"]
    text = MASK_PREFIX_TEXT if processor_readout(processor) == "mask" else PREFIX_TEXT
    prefix = {"ids": prefix_ids(processor, text, len(paths), image_seq_len), "pixel_values": None,
              "pixel_attention_mask": None, "raw_images": None, "n_images": len(paths)}
    it = build_vlm_inputs(processor, state, ex["q"], None, 256, option_order=order, prefix=prefix)
    it["target"] = [ex["target"][i] for i in order]
    it["label"] = max(range(k), key=lambda j: it["target"][j])
    it["qtype"] = vt.QTYPES[ex["q"]["t"]]
    it["order"] = order
    if ex.get("value") is not None:
        it["value"] = float(ex["value"])
    if ex.get("next_target") is not None:
        nt = [float(p) for p in ex["next_target"]]
        it["next_target"] = [nt[i] / sum(nt) for i in order]
    it["feats"] = cache.get(paths) if paths else None
    return it


def _feats(items):
    fs = [it["feats"] for it in items if it.get("feats") is not None]
    return torch.cat(fs) if fs else None


def collate_train(items, pad_id):
    b = collate_vlm(items, pad_id, with_pixels=False)
    b["dataset"] = [it["dataset"] for it in items]
    b["image_hidden_states"] = _feats(items)
    return b


def collate_eval(items, pad_id):
    b = collate_vlm(items, pad_id, with_pixels=False)
    b["index"] = [it["index"] for it in items]
    b["order"] = [it["order"] for it in items]
    b["ids_sha256"] = [vt.input_ids_sha256([it["ids"]]) for it in items]
    b["image_hidden_states"] = _feats(items)
    return b


def forward(model, b):
    if b.get("image_hidden_states") is None:
        return _ORIG["forward"](model, b)
    return model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
                 image_hidden_states=b["image_hidden_states"], option_span=b["option_span"])


# --- sampling -------------------------------------------------------------------------------------------------


def epoch_order(n: int, seed: int, pass_idx: int, worker: int, n_workers: int) -> List[int]:
    """This worker's share of one shuffled pass over ``n`` examples; the workers' shares partition the pass."""
    perm = list(range(n))
    random.Random(seed * 1_000_003 + pass_idx).shuffle(perm)
    return perm[worker::n_workers]


class CachedItemStream(_ORIG["ItemStream"]):
    """laya's ``ItemStream`` (same group weights and ``max_passes`` caps), but each group is drawn as shuffled
    passes without replacement, split across loader workers; items come from the feature cache when given."""

    def __init__(self, processor, examples, seed=0, cache: Optional[FeatureCache] = None, image_seq_len: int = 64,
                 **kw):
        super().__init__(processor, examples, seed, **kw)
        self.cache, self.image_seq_len = cache, image_seq_len

    def _item(self, ex, rng):
        if self.cache is None:
            return vt.make_item(self.processor, ex, rng, **self.item_kw)
        return cached_item(self.processor, ex, rng, self.cache, self.image_seq_len)

    def __iter__(self):
        import math

        wi = torch.utils.data.get_worker_info()
        nw, w = (wi.num_workers, wi.id) if wi else (1, 0)
        rng = random.Random(self.seed * 1000 + w)
        left = {k: (self.max_passes * len(g) / nw if self.max_passes else math.inf) for k, g in self.groups.items()}
        for k in left:
            left[k] -= self.consumed.get(k, 0) / nw
        queue = {k: [] for k in self.keys}
        passes = {k: 0 for k in self.keys}
        while True:
            keys = [k for k in self.keys if left[k] >= 1]
            if not keys:
                return
            key = rng.choices(keys, weights=[self.weights[k] for k in keys])[0]
            left[key] -= 1
            group = self.groups[key]
            if not queue[key]:
                queue[key] = epoch_order(len(group), self.seed * 1000 + self.keys.index(key), passes[key], w, nw)[::-1]
                passes[key] += 1
            # a group smaller than the worker count leaves some workers an empty share: draw at random there
            ex = group[queue[key].pop()] if queue[key] else rng.choice(group)
            try:
                it = self._item(ex, rng)
            except (OSError, ValueError) as e:
                print("skipping %s: %s" % (ex.get("id"), e))
                continue
            it["dataset"] = key
            yield it


class CachedEvalItems(vt._EvalItems):
    def __init__(self, processor, examples, pairs, transform=None, cache=None, image_seq_len=64):
        super().__init__(processor, examples, pairs, transform)
        self.cache, self.image_seq_len = cache, image_seq_len

    def __getitem__(self, j):
        if self.cache is None or self.transform is not None:
            return super().__getitem__(j)
        i, order = self.pairs[j]
        it = cached_item(self.processor, self.examples[i], random.Random(0), self.cache, self.image_seq_len,
                         shuffle=False, order=order)
        it["index"] = i
        return it


def install(cache: Optional[FeatureCache], image_seq_len: int) -> None:
    """Point ``laya.vlm_train.train`` / ``collect_logits`` at the cached items (``cache=None``: sampling only)."""
    vt.ItemStream = functools.partial(CachedItemStream, cache=cache, image_seq_len=image_seq_len)
    if cache is None:
        return
    vt._collate_train = collate_train
    vt._EvalItems = functools.partial(CachedEvalItems, cache=cache, image_seq_len=image_seq_len)
    vt._collate_eval = collate_eval
    vt._forward = forward
