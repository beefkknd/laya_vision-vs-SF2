"""Feasibility probe (docs/laya_text_only_plan.md step 1): cut each fighter's sprite pixels out WITHOUT the background.

Two headless Mesens run the same savestate with the same inputs in lockstep:
  A: normal render (the screen the eye sees), plus a VIDEO dump per captured frame: OAM (544 B), CGRAM (512 B),
     VRAM (64 KiB) and the PPU's OBJ registers from emu.getState().
  B: the same, with --snes.hideBgLayer1..4=true (Mesen's own layer toggles): sprites only, on the backdrop colour.
From A's OAM/VRAM/CGRAM the sprites are re-rendered in Python with each pixel tagged by its OAM index, so every pixel
is owned (P1 / P2 / projectile / other) and checked against B's sprites-only screen.

    .venv/bin/python scripts/probe_sprite_cut.py --state states/vs_ryu_vs_ken.state --frames 40

Writes out/sprite_probe/*.png and summary.json. Uses its own bridge copy (out/sprite_probe/bridge_<port>.lua) with
one extra command, VIDEO; mesen/sf2_bridge.lua is not touched.
"""
import argparse
import json
import os
import re
import time

import numpy as np
from PIL import Image

import _path  # noqa: F401
from sf2.emu.headless import BRIDGE, find_mesen
from sf2.emu.mesen import MesenBridge
from sf2.emu.vs import NAMES, VARS
from sf2.config import DEFAULT_ROM, REPO

OUT = os.path.join(REPO, "out", "sprite_probe")
PORTS = (52011, 52012)
SNAP_EVENT = os.environ.get("PROBE_SNAP_EVENT", "startFrame")
SNAP_VRAM_FROM = 0xC000  # bytes; 0xC000 would cover only the OBJ tables of this game (oamBaseAddress 24576 words)
SIZES = {0: ((8, 8), (16, 16)), 1: ((8, 8), (32, 32)), 2: ((8, 8), (64, 64)), 3: ((16, 16), (32, 32)),
         4: ((16, 16), (64, 64)), 5: ((32, 32), (64, 64)), 6: ((16, 32), (32, 64)), 7: ((16, 32), (32, 32))}

VIDEO_LUA = r'''
    elseif op == "VIDEO" then
      local function blob(mt, size)
        local parts, chunk = {}, {}
        for a = 0, size - 1 do
          chunk[#chunk + 1] = emu.read(a, mt, false)
          if #chunk == 4096 then parts[#parts + 1] = string.char(table.unpack(chunk)); chunk = {} end
        end
        if #chunk > 0 then parts[#parts + 1] = string.char(table.unpack(chunk)) end
        return table.concat(parts)
      end
      local t0 = os.clock()
      local oam, cg, vr
      if cmd[2] == "2" then           -- snapshot taken at endFrame: the OAM/CGRAM/VRAM the grabbed screen used
        oam, cg, vr = snapOam, snapCg, snapVr
      else
        oam = blob(emu.memType.snesSpriteRam, 544)
        cg = blob(emu.memType.snesCgRam, 512)
        vr = (cmd[2] == "1") and blob(emu.memType.snesVideoRam, 65536) or ""
      end
      local t1 = os.clock()
      local st = emu.getState()
      local lines = {}
      for key, v in pairs(st) do
        if string.find(key, "ppu", 1, true) and type(v) ~= "table" then lines[#lines + 1] = key .. "=" .. tostring(v) end
      end
      lines[#lines + 1] = "luaReadSec=" .. tostring(t1 - t0)
      local txt = table.concat(lines, "\n")
      send(string.format("VID %d %d %d %d\n", #txt, #oam, #cg, #vr))
      send(txt); send(oam); send(cg); send(vr)
    elseif op == "PING" then'''

SNAP_LUA = r'''
-- probe: snapshot the sprite-relevant video memory at the end of every rendered frame (before the NMI's DMA)
snapOam, snapCg, snapVr = "", "", ""
local function snapBlob(mt, a0, size)
  local parts, chunk = {}, {}
  for a = a0, a0 + size - 1 do
    chunk[#chunk + 1] = emu.read(a, mt, false)
    if #chunk == 4096 then parts[#parts + 1] = string.char(table.unpack(chunk)); chunk = {} end
  end
  if #chunk > 0 then parts[#parts + 1] = string.char(table.unpack(chunk)) end
  return table.concat(parts)
end
emu.addEventCallback(function()
  snapOam = snapBlob(emu.memType.snesSpriteRam, 0, 544)
  snapCg = snapBlob(emu.memType.snesCgRam, 0, 512)
  snapVr = string.rep("\0", SNAP_VRAM_FROM) .. snapBlob(emu.memType.snesVideoRam, SNAP_VRAM_FROM, 65536 - SNAP_VRAM_FROM)
end, emu.eventType[SNAP_EVENT])
'''


def bridge_copy(port: int) -> str:
    src = open(BRIDGE).read()
    src, n = re.subn(r'local HOST, PORT = "127\.0\.0\.1", \d+', 'local HOST, PORT = "127.0.0.1", %d' % port, src)
    src, m = re.subn(r"local EXIT_ON_DISCONNECT = false", "local EXIT_ON_DISCONNECT = true", src)
    src, k = re.subn(r'\n    elseif op == "PING" then', lambda _: VIDEO_LUA, src)
    src += SNAP_LUA.replace("SNAP_VRAM_FROM", str(SNAP_VRAM_FROM)).replace("SNAP_EVENT", repr(SNAP_EVENT))
    if (n, m, k) != (1, 1, 1):
        raise RuntimeError("bridge patch failed %s" % ((n, m, k),))
    path = os.path.join(OUT, "bridge_%d.lua" % port)
    with open(path, "w") as f:
        f.write(src)
    return path


def argv(port: int, hide_bg: bool):
    rom = os.environ.get("SF2_ROM") or os.path.join(REPO, DEFAULT_ROM)
    extra = ["--snes.hideBgLayer%d=true" % i for i in (1, 2, 3, 4)] if hide_bg else []
    return [find_mesen(), "--testrunner", "--timeout=3600", "--snes.disableFrameSkipping=true",
            "--snes.port2.type=SnesController", *extra, rom, bridge_copy(port)]


VIDEO_MODE = int(os.environ.get('PROBE_VIDEO_MODE', '2'))


def video(b: MesenBridge, vram: bool = True, mode: int = None):
    b._send("VIDEO %d" % (int(vram) if mode is None else mode))
    head = b._line().split()
    if head[0] != "VID":
        raise RuntimeError("expected VID, got %r" % head)
    lt, lo, lc, lv = map(int, head[1:])
    txt = b._read(lt).decode()
    ppu = dict(line.split("=", 1) for line in txt.splitlines() if "=" in line)
    return ppu, b._read(lo), b._read(lc), b._read(lv)


def cg_rgb(cg: bytes) -> np.ndarray:
    w = np.frombuffer(cg, "<u2").astype(np.int32)
    rgb = np.stack([(w & 31), (w >> 5) & 31, (w >> 10) & 31], -1)
    return ((rgb << 3) | (rgb >> 2)).astype(np.uint8)


def tile_pixels(vram: bytes, word_addr: int) -> np.ndarray:
    base = (word_addr * 2) & 0xFFFF
    t = np.frombuffer(vram[base:base + 32], np.uint8) if base + 32 <= 65536 else np.zeros(32, np.uint8)
    out = np.zeros((8, 8), np.uint8)
    for r in range(8):
        planes = (t[2 * r], t[2 * r + 1], t[16 + 2 * r], t[17 + 2 * r])
        for c in range(8):
            bit = 7 - c
            out[r, c] = sum(((planes[p] >> bit) & 1) << p for p in range(4))
    return out


def ppu_get(ppu, *names):
    for n in names:
        for k, v in ppu.items():
            if k.lower().endswith(n.lower()):
                return int(float(v))
    raise KeyError("none of %s in PPU state (%s)" % (names, sorted(ppu)[:40]))


def render_oam(oam: bytes, cg: bytes, vram: bytes, ppu) -> tuple:
    """(rgb 256x256x3, owner 256x256 int: OAM index or -1, entries list). Lower OAM index drawn on top."""
    mode = ppu_get(ppu, "oamMode")
    base = ppu_get(ppu, "oamBaseAddress")
    off = ppu_get(ppu, "oamAddressOffset")
    pal = cg_rgb(cg)
    rgb = np.zeros((256, 256, 3), np.uint8)
    owner = np.full((256, 256), -1, np.int32)
    entries = []
    cache = {}
    for i in range(127, -1, -1):
        x, y, tile, attr = oam[4 * i:4 * i + 4]
        hi = (oam[512 + i // 4] >> (2 * (i % 4))) & 3
        x = x | ((hi & 1) << 8)
        x = x - 512 if x >= 256 else x
        w, h = SIZES[mode][hi >> 1]
        if y >= 224 and y + h <= 256:      # parked off-screen
            continue
        palette = (attr >> 1) & 7
        hflip, vflip = bool(attr & 0x40), bool(attr & 0x80)
        table = attr & 1
        entries.append(dict(i=i, x=x, y=y, w=w, h=h, tile=tile | (table << 8), pal=palette, prio=(attr >> 4) & 3,
                            hflip=hflip, vflip=vflip))
        spr = np.zeros((h, w), np.uint8)
        for tr in range(h // 8):
            for tc in range(w // 8):
                t = ((tile & 0xF0) + tr * 16 + ((tile + tc) & 0x0F)) & 0xFF
                addr = base + (off if table else 0) + t * 16
                if addr not in cache:
                    cache[addr] = tile_pixels(vram, addr)
                spr[tr * 8:tr * 8 + 8, tc * 8:tc * 8 + 8] = cache[addr]
        if hflip:
            spr = spr[:, ::-1]
        if vflip:
            spr = spr[::-1]
        for r in range(h):
            yy = (y + r) & 0xFF
            for c in range(w):
                xx = x + c
                if 0 <= xx < 256 and spr[r, c]:
                    rgb[yy, xx] = pal[128 + palette * 16 + spr[r, c]]
                    owner[yy, xx] = i
    return rgb, owner, entries


def sprite_mask(img: np.ndarray) -> np.ndarray:
    """Hide-BG screen: the backdrop is one colour per row (the stage's HDMA sky gradient); anything else = sprite."""
    out = np.zeros(img.shape[:2], bool)
    for y in range(img.shape[0]):
        row = img[y].reshape(-1, 3)
        vals, counts = np.unique(row, axis=0, return_counts=True)
        out[y] = np.any(row != vals[counts.argmax()], -1)
    return out


def best_shift(owner: np.ndarray, smask: np.ndarray):
    best = (0, -1.0)
    for dy in range(-4, 5):
        sh = np.roll(owner >= 0, dy, 0)[:smask.shape[0]]
        iou = (sh & smask).sum() / max((sh | smask).sum(), 1)
        if iou > best[1]:
            best = (dy, float(iou))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="states/vs_ryu_vs_ken.state")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--p2-fireball", action="store_true")
    ap.add_argument("--show", type=int, nargs="*", default=[12, 25, 40])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = os.path.basename(args.state).replace(".state", "")
    state = open(os.path.join(REPO, args.state), "rb").read()
    qcf = [["down"]] * 3 + [["down", "right"]] * 3 + [["right", "y"]] * 2
    p1 = (qcf + [[]] * 12 + [["right"]] * 60)[:args.frames]
    p2 = ([[]] * 10 + [["up", "left"]] * 4 + [[]] * 60)[:args.frames]
    if args.p2_fireball:
        p2 = ([[]] * 4 + [["down"]] * 3 + [["down", "left"]] * 3 + [["left", "y"]] * 2 + [[]] * 80)[:args.frames]
    bridges, per = [], []
    t_run, t_vid = [], []
    try:
        for port, hide in zip(PORTS, (False, True)):
            b = MesenBridge(port, launch=argv(port, hide))
            b.set_capture("raw")
            b.set_vars(VARS)
            b.load_state(state)
            bridges.append(b)
        a, s = bridges
        for f in range(args.frames):
            t0 = time.time()
            oa = a.run([p1[f]], caps=[1], p2=[p2[f]])
            t1 = time.time()
            ppu, oam, cg, vram = video(a, mode=VIDEO_MODE)
            t2 = time.time()
            os_ = s.run([p1[f]], caps=[1], p2=[p2[f]])
            t_run.append(t1 - t0); t_vid.append(t2 - t1)
            per.append(dict(f=f + 1, ram=dict(zip(NAMES, oa.rams[-1])), normal=oa.images[1], hide=os_.images[1],
                            ppu=ppu, oam=oam, cg=cg, vram=vram))
        a.load_state(state)
        t0 = time.time()
        for f in range(args.frames):
            a.run([p1[f]], p2=[p2[f]])
        t_nocap = (time.time() - t0) / args.frames
    finally:
        for b in bridges:
            b.close()

    # OAM dumped at the poll after frame f: does it draw the screen of frame f or f+1?
    lag_iou = {0: [], 1: []}
    frames = []
    for j, d in enumerate(per):
        rgb, owner, entries = render_oam(d["oam"], d["cg"], d["vram"], d["ppu"])
        d.update(rgb=rgb, owner=owner, entries=entries)
        for lag in (0, 1):
            if j + lag < len(per):
                lag_iou[lag].append(best_shift(owner, sprite_mask(per[j + lag]["hide"])))
    lag = max(lag_iou, key=lambda L: np.mean([x[1] for x in lag_iou[L]]))
    best_lag = []
    for j, d in enumerate(per):
        ious = {L: best_shift(d["owner"], sprite_mask(per[j + L]["hide"]))[1]
                for L in (-1, 0, 1, 2) if 0 <= j + L < len(per)}
        best_lag.append((j + 1, max(ious, key=ious.get), round(max(ious.values()), 3)))
    print("per-frame best lag (frame, lag, iou):", best_lag)
    dys = sorted({x[0] for x in lag_iou[lag]})
    bad = [(j + 1, round(x[1], 3)) for j, x in enumerate(lag_iou[lag]) if x[1] < 0.97]
    dy = lag_iou[lag][0][0]
    pal_stats = {}
    for j, d in enumerate(per):
        r = d["ram"]
        groups = {}
        for e in d["entries"]:
            groups.setdefault(e["pal"], []).append(e)
        row = dict(f=d["f"], p1_x=r["p1_x"], p2_x=r["p2_x"], p1_y=r["p1_y"], p2_y=r["p2_y"], shot1=r["shot1"],
                   shot1_x=r["shot1_x"], n_entries=len(d["entries"]))
        for p, es in sorted(groups.items()):
            row["pal%d" % p] = "%d-%d n%d x%d-%d" % (min(e["i"] for e in es), max(e["i"] for e in es), len(es),
                                                   min(e["x"] for e in es), max(e["x"] + e["w"] for e in es))
        frames.append(row)
    colors = {0: (255, 140, 0), 1: (0, 220, 120), 2: (255, 255, 255), 3: (120, 120, 120), 4: (255, 80, 80),
              5: (255, 220, 0), 6: (80, 160, 255), 7: (200, 0, 200)}
    for k in args.show:
        j = k - 1
        if j + lag >= len(per):
            continue
        d, scr = per[j], per[j + lag]
        own = np.roll(d["owner"], dy, 0)[:224]
        rgb = np.roll(d["rgb"], dy, 0)[:224]
        pal_of = {e["i"]: e["pal"] for e in d["entries"]}
        vis = np.zeros((224, 256, 3), np.uint8)
        for i, p in pal_of.items():
            vis[own == i] = colors[p]
        smask = sprite_mask(scr["hide"])
        Image.fromarray(np.concatenate([scr["normal"], scr["hide"], vis, np.repeat(smask[..., None] * 255, 3, -1).astype(np.uint8)], 1)).save(
            os.path.join(OUT, "%s_f%02d_normal_hidebg_owner_mask.png" % (tag, k)))
        for p in sorted(set(pal_of.values())):
            m = np.isin(own, [i for i, q in pal_of.items() if q == p])
            ys, xs = np.nonzero(m)
            if len(ys) == 0:
                continue
            rgba = np.dstack([rgb, np.where(m, 255, 0).astype(np.uint8)])
            crop = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            Image.fromarray(crop).resize((crop.shape[1] * 3, crop.shape[0] * 3), Image.NEAREST).save(
                os.path.join(OUT, "%s_f%02d_cut_pal%d.png" % (tag, k, p)))
        # colour check: OAM render vs hide-BG screen where both have sprite pixels
        both = (own >= 0) & smask
        diff = np.abs(rgb.astype(int) - scr["hide"].astype(int)).max(-1)
        frames[j]["colour_match"] = "%d/%d" % (int(((diff <= 8) & both).sum()), int(both.sum()))
        frames[j]["oam_px_not_in_hide"] = int(((own >= 0) & ~smask).sum())
        frames[j]["hide_px_not_in_oam"] = int((smask & (own < 0)).sum())
    summary = dict(state=args.state, lag=lag, frames_iou_below_0_97=bad,
                   min_iou=round(min(x[1] for x in lag_iou[lag]), 4), dy_values=dys, dy=dy,
                   mean_iou={L: float(np.mean([x[1] for x in v])) for L, v in lag_iou.items()},
                   ppu_obj={kk: v for kk, v in per[0]["ppu"].items() if "oam" in kk.lower() or "obj" in kk.lower()},
                   sec_per_frame_run_with_raw_capture=float(np.median(t_run)),
                   sec_per_frame_video_dump=float(np.median(t_vid)),
                   sec_per_frame_run_nocap=t_nocap, frames=frames)
    with open(os.path.join(OUT, "%s_summary.json" % tag), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}, default=str))
    for row in frames:
        print(row)


if __name__ == "__main__":
    main()
