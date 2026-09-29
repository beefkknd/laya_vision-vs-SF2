"""The part 2 YouTube video: the four recorded parts joined, a part 1 vs part 2 comparison at the end, and captions,
in two versions:

    en   your English voice, English captions
    zh   a Chinese voice (Kokoro, zf_xiaobei) speaking Qwen's translation of each line, Chinese captions

Runs in the media venv (.venv-media: mlx-whisper, mlx-audio/Kokoro, Pillow); Qwen on omlx translates.

    .venv-media/bin/python tools/make_video.py                 # everything, from ~/Desktop/laya_video/work
    .venv-media/bin/python tools/make_video.py --only en

Inputs (made by earlier steps): stitched_en_nocaptions.mp4 (the four parts joined), work/transcript_en.json (Whisper
large-v3-turbo). Outputs in ~/Desktop/laya_video/: laya_sf2_part2_en.mp4 / _zh.mp4 (captions burned in) and
captions_en.srt / captions_zh.srt (to upload to YouTube as captions). ffmpeg here has no text filters, so captions
are drawn with Pillow and overlaid as one transparent image track.
"""
import argparse
import json
import os
import re
import subprocess
import urllib.request
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.expanduser("~/Desktop/laya_video")
WORK = os.path.join(OUT, "work")
STITCHED = os.path.join(OUT, "stitched_en_nocaptions.mp4")
CROP_TOP = 94                                 # the browser's tabs and address bar at the top of the recording
W, H = 1420, 1150 - CROP_TOP
SR = 24000                                   # Kokoro's sample rate
QWEN = "http://127.0.0.1:8000/v1/chat/completions"
QWEN_MODEL = "Jundot--Qwen3.8-27B-oQ4e-mtp"
FONT_ZH = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FONT_EN = "/System/Library/Fonts/Hiragino Sans GB.ttc"     # has the arrow and every Latin glyph the captions use

# what Whisper mishears in the recording -> the real names (only names; the wording stays as spoken)
FIXES = [
    (r"\b(?:[Ll]iar|[Ll]ive|[Ll]ight|[Ll]aya?|[Ll]ayer)[ -]?[Vv](?:ision|ersion)\b", "laya-vision"),
    (r"\b[Ll]iar[Vv]ision\b", "laya-vision"),
    (r"\b(?:latex|[Tt]ext) layer\b", "text laya"),
    (r"\b(?:[Ll]ayer|[Ll]iar|[Ll]aya) text\b", "text laya"),
    (r"\bQueen's\b|\bqueen's\b", "Qwen's"),
    (r"\b[Qq]ueen\b|\b[Qq]uinn\b", "Qwen"),
    (r"\bmy laura\b", "my LoRA"),
    (r"\b[Ss]treet [Ff]ight\b(?!er)", "Street Fighter"),
    (r"\bchunli\b", "Chun-Li"),
    (r"\breal liar move\b", "real laya move"),
]

COMPARE_EN = [
    "Before I wrap up, here is how this version compares with part one.",
    "Part one's laya-vision was trained for one job: beating Dhalsim.",
    "After imitation and two rounds of DAgger, it won forty of forty-one rounds against him.",
    "But it was rigid. It only knew Dhalsim,",
    "and changing how it played meant new data and hours of fine-tuning.",
    "The new system splits the work.",
    "laya-vision sees the screen, text laya follows plain-language advice,",
    "and Qwen writes that advice between rounds.",
    "So the play can change without retraining.",
    "It is not as strong yet, and it still needs fine-tuning,",
    "but now Qwen can enhance the game play as it goes.",
]
COMPARE_ZH = [
    "最后，我们把这个版本和第一部分对比一下。",
    "第一部分的 laya-vision 只为一件事训练：打败达尔锡。",
    "经过模仿学习和两轮 DAgger，它对达尔锡四十一个回合赢了四十个。",
    "但它很死板。它只认识达尔锡，",
    "想改变它的打法，就要重新收集数据，再花几个小时微调。",
    "新的系统把工作分开了。",
    "laya-vision 看屏幕，text laya 听懂白话建议，",
    "Qwen 在回合之间写这些建议。",
    "所以打法可以随时改变，不用重新训练。",
    "它现在还没那么强，也还需要微调，",
    "但 Qwen 已经能在对局中一边打一边提升玩法了。",
]
SLIDE = {
    "en": {"title": "Part 1 vs Part 2",
           "left": ("Part 1 · one model, one job", ["laya-vision alone, 14 moves", "imitation + 2 rounds of DAgger",
                                                    "won 40 of 41 rounds vs Dhalsim", "only knew Dhalsim",
                                                    "new play = new data + hours of fine-tuning"], "RIGID"),
           "right": ("Part 2 · two systems", ["laya-vision sees, text laya follows advice", "Qwen coaches between rounds",
                                              "the whole arcade ladder", "play changes with no retraining",
                                              "still needs fine-tuning; Qwen still learning"], "FLEXIBLE"),
           "foot": "Strong but rigid  →  flexible, and Qwen can enhance the play"},
    "zh": {"title": "第一部分 vs 第二部分",
           "left": ("第一部分 · 一个模型，一个任务", ["只有 laya-vision，14 个招式", "模仿学习 + 两轮 DAgger",
                                                "对达尔锡 41 回合赢 40", "只认识达尔锡", "改打法 = 新数据 + 数小时微调"], "死板"),
           "right": ("第二部分 · 两个系统", ["laya-vision 看，text laya 听建议", "Qwen 在回合之间当教练",
                                        "打完整个街机关卡", "打法随时改变，不用重新训练", "还需要微调；Qwen 还在学习"], "灵活"),
           "foot": "强但死板  →  灵活，而且 Qwen 能提升玩法"},
}


def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


# ---- captions --------------------------------------------------------------------------------------------------

def fix(text: str) -> str:
    for pat, rep in FIXES:
        text = re.sub(pat, rep, text)
    return re.sub(r"\s+", " ", text).strip()


def cues_en(transcript: Dict, max_chars: int = 84) -> List[Tuple[float, float, str]]:
    """Caption cues from Whisper's word timings: at most ``max_chars`` per cue (two lines), names fixed."""
    cues = []
    for seg in transcript["segments"]:
        words, cur, t0 = seg.get("words") or [], [], None
        for w in words:
            if t0 is None:
                t0 = w["start"]
            cur.append(w)
            text = "".join(x["word"] for x in cur)
            if len(text) > max_chars or w["word"].rstrip().endswith((".", "?", "!")) and len(text) > 30:
                cues.append((t0, w["end"], fix(text)))
                cur, t0 = [], None
        if cur:
            cues.append((t0, cur[-1]["end"], fix("".join(x["word"] for x in cur))))
    return [c for c in cues if c[2]]


def srt(cues: List[Tuple[float, float, str]], path: str) -> None:
    def ts(t):
        ms = int(round(t * 1000))
        return "%02d:%02d:%02d,%03d" % (ms // 3600000, ms // 60000 % 60, ms // 1000 % 60, ms % 1000)
    with open(path, "w") as f:
        for i, (a, b, text) in enumerate(cues, 1):
            f.write("%d\n%s --> %s\n%s\n\n" % (i, ts(a), ts(b), text))


NAMES = ("Keep these names as written: laya-vision, text laya, Qwen, LoRA, DAgger, Mesen, omlx. Chun-Li = 春丽, "
         "Dhalsim = 达尔锡, Ryu = 隆, Ken = 肯, Street Fighter = 街头霸王.")


def _qwen(prompt: str) -> str:
    body = {"model": QWEN_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 8192,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(QWEN, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def translate(lines: List[str]) -> List[str]:
    """Qwen translates caption lines to simplified Chinese: 25 at a time; a batch whose reply is not a list of the
    right length is translated one line at a time instead."""
    out = []
    for i in range(0, len(lines), 25):
        batch = lines[i:i + 25]
        try:
            text = _qwen("Translate each line of this spoken video narration into natural, spoken simplified Chinese. "
                         "%s Reply with a JSON list of exactly %d strings, one per line, in order, nothing else.\n\n%s"
                         % (NAMES, len(batch), json.dumps(batch, ensure_ascii=False)))
            got = json.loads(text[text.find("["): text.rfind("]") + 1])
            if len(got) != len(batch):
                raise ValueError("%d lines for %d" % (len(got), len(batch)))
        except ValueError as e:
            print("batch %d: %s; one line at a time" % (i, e), flush=True)
            got = [_qwen("Translate this spoken line into natural, spoken simplified Chinese. %s Reply with the "
                         "Chinese only.\n\n%s" % (NAMES, line)).strip() for line in batch]
        out += [str(x).strip() for x in got]
        print("translated %d/%d" % (len(out), len(lines)), flush=True)
    return out


# ---- caption track: transparent images overlaid on the video ---------------------------------------------------

def caption_png(text: str, lang: str, path: str) -> None:
    font = ImageFont.truetype(FONT_ZH if lang == "zh" else FONT_EN, 40 if lang == "zh" else 38)
    img = Image.new("RGBA", (W, 170), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lines, cur = [], ""
    for ch in (list(text) if lang == "zh" else re.split(r"(\s+)", text)):
        if d.textlength(cur + ch, font=font) > W - 160 and cur.strip():
            lines.append(cur.strip())
            cur = ch.lstrip()
        else:
            cur += ch
    lines = (lines + [cur.strip()])[-2:]
    y = 170 - 20 - len(lines) * 56
    for line in lines:
        tw = d.textlength(line, font=font)
        box = (W / 2 - tw / 2 - 16, y - 6, W / 2 + tw / 2 + 16, y + 50)
        d.rounded_rectangle(box, 10, fill=(0, 0, 0, 170))
        d.text((W / 2 - tw / 2, y), line, font=font, fill=(255, 255, 255, 255))
        y += 56
    img.save(path)


def caption_track(cues: List[Tuple[float, float, str]], lang: str, total: float, tag: str) -> str:
    """A concat list of caption images (blank between cues) covering ``total`` seconds; returns its path."""
    d = os.path.join(WORK, "caps_%s" % tag)
    os.makedirs(d, exist_ok=True)
    blank = os.path.join(d, "blank.png")
    Image.new("RGBA", (W, 170), (0, 0, 0, 0)).save(blank)
    lines, t = ["ffconcat version 1.0"], 0.0
    for i, (a, b, text) in enumerate(cues):
        a, b = max(a, t), min(max(b, a + 0.8), cues[i + 1][0] if i + 1 < len(cues) else total)
        if b <= a:
            continue
        if a > t:
            lines += ["file '%s'" % blank, "duration %.3f" % (a - t)]
        png = os.path.join(d, "c%04d.png" % i)
        caption_png(text, lang, png)
        lines += ["file '%s'" % png, "duration %.3f" % (b - a)]
        t = b
    lines += ["file '%s'" % blank, "duration %.3f" % max(0.1, total - t), "file '%s'" % blank]
    path = os.path.join(d, "track.ffconcat")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ---- voice --------------------------------------------------------------------------------------------------------

_KOKORO = None


def speak(text: str, lang: str) -> np.ndarray:
    global _KOKORO
    if _KOKORO is None:
        from mlx_audio.tts.utils import load_model
        _KOKORO = load_model("prince-canuma/Kokoro-82M")
    voice, code = ("zf_xiaobei", "z") if lang == "zh" else ("af_heart", "a")
    return np.concatenate([np.array(r.audio).reshape(-1) for r in _KOKORO.generate(text=text, voice=voice,
                                                                                    lang_code=code, speed=1.0)])


def fit(audio: np.ndarray, seconds: float) -> np.ndarray:
    """Speed a clip up (never slower) so it fits ``seconds``, at most 1.6x; beyond that it runs over."""
    if len(audio) / SR <= seconds:
        return audio
    rate = min(1.6, len(audio) / SR / seconds)
    src, dst = os.path.join(WORK, "fit_in.wav"), os.path.join(WORK, "fit_out.wav")
    sf.write(src, audio, SR)
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-filter:a", "atempo=%.4f" % rate, dst])
    return sf.read(dst)[0]


def voice_track(cues: List[Tuple[float, float, str]], total: float, lang: str, path: str) -> None:
    """Each cue spoken at its start time, sped up to end before the next cue starts."""
    buf = np.zeros(int((total + 2) * SR))
    for i, (a, b, text) in enumerate(cues):
        slot = (cues[i + 1][0] if i + 1 < len(cues) else total) - a
        clip = fit(speak(text, lang), max(0.5, slot - 0.05))
        s = int(a * SR)
        buf[s:s + len(clip)] += clip[: max(0, len(buf) - s)]
        if i % 20 == 0:
            print("voice %d/%d" % (i, len(cues)), flush=True)
    sf.write(path, np.clip(buf[: int(total * SR)], -1, 1), SR)


# ---- the comparison segment -------------------------------------------------------------------------------------

def slide(lang: str, path: str) -> None:
    s = SLIDE[lang]
    font = FONT_ZH if lang == "zh" else FONT_EN
    f = lambda n: ImageFont.truetype(font, n)
    img = Image.new("RGB", (W, H), (15, 20, 40))
    d = ImageDraw.Draw(img)
    d.text((80, 40), s["title"], font=f(60), fill=(232, 236, 248))     # all above the caption area (bottom ~210 px)
    bw = 630                                          # box width
    for x, (head, items, tag), col in ((60, s["left"], (142, 152, 189)), (730, s["right"], (91, 157, 255))):
        d.rounded_rectangle((x, 140, x + bw, 760), 22, fill=(22, 29, 56), outline=col, width=3)
        d.text((x + 34, 172), head, font=f(32), fill=(232, 236, 248))
        y = 245
        for it in items:
            size = 27
            while d.textlength(it, font=f(size)) > bw - 100 and size > 18:      # every line fits its box
                size -= 1
            d.ellipse((x + 38, y + 14, x + 50, y + 26), fill=col)
            d.text((x + 66, y), it, font=f(size), fill=(210, 216, 235))
            y += 82
        tw = d.textlength(tag, font=f(40))
        d.text((x + bw / 2 - tw / 2, 680), tag, font=f(40), fill=col)
    tw = d.textlength(s["foot"], font=f(36))
    d.text((W / 2 - tw / 2, 790), s["foot"], font=f(36), fill=(240, 163, 58))
    img.save(path)


def compare_segment(lang: str) -> Tuple[str, List[Tuple[float, float, str]]]:
    """The comparison clip (slide + narration) and its caption cues (times from 0)."""
    lines = COMPARE_ZH if lang == "zh" else COMPARE_EN
    parts, cues, t = [], [], 0.6
    for line in lines:
        a = speak(line, lang)
        cues.append((t, t + len(a) / SR, line))
        parts += [a, np.zeros(int(0.35 * SR))]
        t += len(a) / SR + 0.35
    audio = np.concatenate([np.zeros(int(0.6 * SR))] + parts + [np.zeros(int(1.5 * SR))])
    wav, png, mp4 = (os.path.join(WORK, "compare_%s.%s" % (lang, e)) for e in ("wav", "png", "mp4"))
    sf.write(wav, audio, SR)
    slide(lang, png)
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-framerate", "60000/1001", "-i", png,
         "-i", wav, "-c:v", "h264_videotoolbox", "-b:v", "4M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-ar", "48000", "-ac", "2", "-shortest", mp4])
    return mp4, cues


# ---- assembling ---------------------------------------------------------------------------------------------------

def build(lang: str, main_cues: List[Tuple[float, float, str]]) -> None:
    main_len = duration(STITCHED)
    comp_mp4, comp_cues = compare_segment(lang)
    total = main_len + duration(comp_mp4)
    cues = main_cues + [(a + main_len, b + main_len, t) for a, b, t in comp_cues]
    srt(cues, os.path.join(OUT, "captions_%s.srt" % lang))
    track = caption_track(cues, lang, total, lang)
    audio_in = STITCHED
    if lang == "zh":                                   # the Chinese voice replaces the English one
        wav = os.path.join(WORK, "voice_zh.wav")
        voice_track(main_cues, main_len, "zh", wav)
        audio_in = wav
    out = os.path.join(OUT, "laya_sf2_part2_%s.mp4" % lang)
    stereo = "aresample=48000,aformat=sample_rates=48000:channel_layouts=stereo"
    main = "[0:v]crop=%d:%d:0:%d,setsar=1[m];" % (W, H, CROP_TOP)
    if lang == "en":        # inputs: 0 the four parts, 1 the comparison, 2 the caption images
        inputs = ["-i", STITCHED, "-i", comp_mp4]
        graph = (main + "[1:v]setsar=1[c];[0:a]%s[ma];[1:a]%s[ca];"
                 "[m][ma][c][ca]concat=n=2:v=1:a=1[v0][a];[v0][2:v]overlay=0:H-h-40[v]" % (stereo, stereo))
    else:                   # inputs: 0 the four parts (picture only), 1 the comparison, 2 the Chinese voice, 3 captions
        inputs = ["-i", STITCHED, "-i", comp_mp4, "-i", audio_in]
        graph = (main + "[1:v]setsar=1[c];[m][c]concat=n=2:v=1:a=0[v0];"
                 "[2:a]%s,apad=whole_dur=%.3f,atrim=0:%.3f[za];[1:a]%s[ca];[za][ca]concat=n=2:v=0:a=1[a];"
                 "[v0][3:v]overlay=0:H-h-40[v]" % (stereo, main_len, main_len, stereo))
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + inputs +
        ["-f", "concat", "-safe", "0", "-i", track, "-filter_complex", graph,
         "-map", "[v]", "-map", "[a]", "-c:v", "h264_videotoolbox", "-b:v", "6M", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out])
    print("wrote %s (%.0f s)" % (out, duration(out)), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("en", "zh"))
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    en = cues_en(json.load(open(os.path.join(WORK, "transcript_en.json"))))
    if args.only != "zh":
        build("en", en)
    if args.only != "en":
        zh_path = os.path.join(WORK, "cues_zh.json")
        if os.path.exists(zh_path):
            zh_text = json.load(open(zh_path))
        else:
            zh_text = translate([t for _, _, t in en])
            json.dump(zh_text, open(zh_path, "w"), ensure_ascii=False, indent=1)
        build("zh", [(a, b, z) for (a, b, _), z in zip(en, zh_text)])


if __name__ == "__main__":
    main()
