#!/usr/bin/env python3
"""
GTA 6 Streamer Clipper
----------------------
Pulls a stream VOD (Twitch / YouTube / Kick URL or a local file), finds the hype
moments from audio loudness spikes + chat activity, and exports vertical 9:16
clips ready for TikTok / Shorts / Reels, with optional burned-in captions.

    python clipper.py "https://www.twitch.tv/videos/123456789" --top 8
    python clipper.py vod.mp4 --chat chat.json --layout facecam --facecam 1500,40,380,215
    python clipper.py vod.mp4 --dry-run          # just list the moments

Only clip streamers who allow it (most have a clipping/affiliate policy).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Chat words that usually mean "something just happened"
HYPE_PATTERNS = [
    r"\bclip (it|that)\b", r"\bkekw\b", r"\bomegalul\b", r"\blul\b", r"\blmf?ao+\b",
    r"\bno way\b", r"\bw+\b", r"\bpog(gers|champ)?\b", r"\bwtf\b", r"\bholy\b",
    r"\?{3,}", r"\bbro+\b", r"\bnah+\b", r"\bhuh\b", r"\bdead\b", r"💀", r"😂", r"🤣",
    r"\bicant\b", r"\bsheesh\b", r"\bcinema\b", r"\bpeak\b", r"\bwasted\b",
]
HYPE_RE = re.compile("|".join(HYPE_PATTERNS), re.IGNORECASE)


@dataclass
class Moment:
    rank: int
    peak: float
    start: float
    end: float
    score: float
    audio_z: float
    chat_z: float
    chat_msgs: int


# ---------------------------------------------------------------- utilities
def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


def need(binary: str) -> None:
    if not shutil.which(binary):
        sys.exit(f"error: '{binary}' not found on PATH")


def hms(t: float) -> str:
    t = int(t)
    return f"{t // 3600:d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def duration_of(path: Path) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    return float(out.stdout.strip())


def robust_z(x: np.ndarray, window: int) -> np.ndarray:
    """z-score each second against a rolling median/MAD, so a loud stream
    doesn't make everything 'hype' and a quiet one still finds its peaks."""
    if len(x) == 0:
        return x
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(xp, window)[: len(x)]
    med = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - med[:, None]), axis=1) * 1.4826
    mad = np.maximum(mad, max(np.std(x) * 0.25, 1e-6))
    return (x - med) / mad


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) == 0:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


# ---------------------------------------------------------------- download
def download(url: str, workdir: Path, max_height: int, want_chat: bool) -> tuple[Path, Path | None]:
    need("yt-dlp")
    workdir.mkdir(parents=True, exist_ok=True)
    tmpl = str(workdir / "vod.%(ext)s")
    print(f"[download] {url}")
    run(["yt-dlp", "-f", f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
         "--merge-output-format", "mp4", "-o", tmpl, "--no-playlist", url])
    video = next(p for p in workdir.glob("vod.*") if p.suffix in {".mp4", ".mkv", ".webm"})

    chat_path = None
    if want_chat:
        chat_path = workdir / "chat.json"
        try:
            fetch_chat(url, chat_path)
        except Exception as e:  # chat is a bonus, never fatal
            print(f"[chat] could not fetch chat ({e}); continuing with audio only")
            chat_path = None
    return video, chat_path


def fetch_chat(url: str, out: Path) -> None:
    """Twitch/YouTube/Kick chat replay via chat-downloader; YouTube fallback via yt-dlp."""
    try:
        from chat_downloader import ChatDownloader  # type: ignore
        print("[chat] downloading chat replay (this can take a while on long VODs)")
        rows = []
        for m in ChatDownloader().get_chat(url, message_types=["text_message"]):
            t = m.get("time_in_seconds")
            if t is not None:
                rows.append({"t": float(t), "msg": m.get("message", "")})
        out.write_text(json.dumps(rows))
        print(f"[chat] {len(rows)} messages")
        return
    except ImportError:
        pass
    # yt-dlp can save YouTube live chat replay as a subtitle track
    run(["yt-dlp", "--skip-download", "--write-subs", "--sub-langs", "live_chat",
         "-o", str(out.with_suffix("")), url])
    produced = out.with_suffix(".live_chat.json")
    if not produced.exists():
        raise RuntimeError("no chat replay available")
    produced.rename(out)


def load_chat(path: Path) -> list[tuple[float, str]]:
    """Accepts: our own [{t, msg}] JSON, chat-downloader JSON, yt-dlp live_chat
    JSON-lines, or a CSV with columns time/t/seconds + message/msg/text."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    msgs: list[tuple[float, str]] = []

    if path.suffix.lower() == ".csv":
        for row in csv.DictReader(text.splitlines()):
            t = row.get("t") or row.get("time") or row.get("seconds") or row.get("time_in_seconds")
            m = row.get("msg") or row.get("message") or row.get("text") or ""
            if t:
                msgs.append((parse_time(t), m))
        return msgs

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("comments") or data.get("messages") or []
        for d in data:
            if "t" in d:
                msgs.append((float(d["t"]), d.get("msg", "")))
            elif "time_in_seconds" in d:
                msgs.append((float(d["time_in_seconds"]), d.get("message", "")))
            elif "content_offset_seconds" in d:  # Twitch API / TwitchDownloader
                body = d.get("message", {})
                msgs.append((float(d["content_offset_seconds"]),
                             body.get("body", "") if isinstance(body, dict) else str(body)))
        return msgs
    except json.JSONDecodeError:
        pass

    for line in text.splitlines():  # yt-dlp live_chat: JSON lines
        try:
            d = json.loads(line)
            r = d["replayChatItemAction"]
            t = int(r["videoOffsetTimeMsec"]) / 1000
            for a in r["actions"]:
                item = a.get("addChatItemAction", {}).get("item", {})
                rend = item.get("liveChatTextMessageRenderer")
                if rend:
                    runs = rend.get("message", {}).get("runs", [])
                    msgs.append((t, "".join(x.get("text", "") for x in runs)))
        except (KeyError, ValueError, TypeError):
            continue
    return msgs


def parse_time(s: str) -> float:
    s = s.strip()
    if ":" in s:
        sec = 0.0
        for part in s.split(":"):
            sec = sec * 60 + float(part)
        return sec
    return float(s)


# ---------------------------------------------------------------- signals
def audio_loudness(video: Path, dur: float) -> np.ndarray:
    """Per-second loudness in dB, decoded at 8 kHz mono (fast even on 8h VODs)."""
    print("[audio] analysing loudness")
    sr = 8000
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1",
                             "-ar", str(sr), "-f", "s16le", "-"], stdout=subprocess.PIPE)
    n = int(dur) + 1
    db = np.full(n, -90.0)
    i = 0
    while True:
        chunk = proc.stdout.read(sr * 2)
        if not chunk:
            break
        a = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768
        if i < n:
            db[i] = 20 * np.log10(np.sqrt(np.mean(a * a)) + 1e-9)
        i += 1
    proc.wait()
    return db[: max(i, 1)]


def chat_signal(msgs: list[tuple[float, str]], n: int, delay: float) -> tuple[np.ndarray, np.ndarray]:
    """Messages/sec and hype-weighted messages/sec, shifted back by `delay`
    because chat reacts a few seconds after the moment happens."""
    rate = np.zeros(n)
    hype = np.zeros(n)
    for t, m in msgs:
        i = int(t - delay)
        if 0 <= i < n:
            rate[i] += 1
            if HYPE_RE.search(m or ""):
                hype[i] += 1
    return rate, hype


def find_moments(audio_db: np.ndarray, chat: tuple[np.ndarray, np.ndarray] | None,
                 top: int, pre: float, post: float, min_gap: float, dur: float,
                 audio_weight: float, min_score: float = 2.0) -> list[Moment]:
    n = len(audio_db)
    az = smooth(robust_z(audio_db, 301), 5)  # 5-min baseline, 5s smoothing
    if chat is not None and chat[0].sum() > 0:
        rate, hype = chat
        cz = smooth(robust_z(smooth(rate + 2 * hype, 3), 301), 7)
        score = audio_weight * az + (1 - audio_weight) * cz
    else:
        cz = np.zeros(n)
        score = az

    order = np.argsort(score)[::-1]
    picked: list[int] = []
    for i in order:
        if score[i] < min_score or len(picked) >= top:
            break
        if all(abs(i - p) >= min_gap for p in picked):
            picked.append(int(i))

    moments = []
    for i in sorted(picked, key=lambda p: -score[p]):
        start = max(0.0, i - pre)
        end = min(dur, i + post)
        msgs = int(chat[0][int(start):int(end)].sum()) if chat is not None else 0
        moments.append(Moment(0, float(i), start, end, float(score[i]),
                              float(az[i]), float(cz[i]), msgs))
    for r, m in enumerate(moments, 1):
        m.rank = r
    return moments


# ---------------------------------------------------------------- captions
def transcribe(video: Path, m: Moment, model_name: str, out: Path) -> Path | None:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        print("[captions] faster-whisper not installed; skipping (pip install faster-whisper)")
        return None
    global _WHISPER
    if "_WHISPER" not in globals():
        print(f"[captions] loading whisper '{model_name}'")
        _WHISPER = WhisperModel(model_name, device="auto", compute_type="int8")
    wav = out.with_suffix(".wav")
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{m.start:.2f}", "-t", f"{m.end - m.start:.2f}",
         "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    segs, _ = _WHISPER.transcribe(str(wav), word_timestamps=True, vad_filter=True)
    words = [w for s in segs for w in (s.words or [])]
    wav.unlink(missing_ok=True)
    if not words:
        return None
    lines, idx = [], 1
    for k in range(0, len(words), 3):  # 3-word punchy caption chunks
        chunk = words[k:k + 3]
        lines.append(f"{idx}\n{srt_ts(chunk[0].start)} --> {srt_ts(chunk[-1].end)}\n"
                     f"{' '.join(w.word.strip() for w in chunk).upper()}\n")
        idx += 1
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def srt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


# ---------------------------------------------------------------- render
def vertical_filter(layout: str, facecam: tuple[int, int, int, int] | None) -> str:
    if layout == "crop":
        return "[0:v]crop=trunc(ih*9/16/2)*2:ih,scale=1080:1920,setsar=1[v]"
    if layout == "facecam":
        x, y, w, h = facecam  # type: ignore[misc]
        return (f"[0:v]split=2[a][b];"
                f"[a]crop={w}:{h}:{x}:{y},scale=1080:640:force_original_aspect_ratio=increase,"
                f"crop=1080:640[cam];"
                f"[b]crop=trunc(ih*27/32/2)*2:ih,scale=1080:1280[game];"
                f"[cam][game]vstack,setsar=1[v]")
    # default "blur": full gameplay in the middle, blurred copy filling the frame
    return ("[0:v]split=2[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
            "boxblur=20:2,eq=brightness=-0.08[bgb];"
            "[fg]scale=1080:-2[fgs];[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1[v]")


def render(video: Path, m: Moment, out: Path, layout: str, facecam, srt: Path | None,
           title: str | None, crf: int) -> None:
    fc = vertical_filter(layout, facecam)
    last = "[v]"
    if srt is not None:
        style = ("FontName=Arial,FontSize=16,Bold=1,PrimaryColour=&H00FFFFFF,"
                 "OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=0,"
                 "Alignment=2,MarginV=70")
        p = str(srt).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        fc += f";{last}subtitles='{p}':force_style='{style}'[vs]"
        last = "[vs]"
    if title:
        t = title.replace("\\", "\\\\").replace("'", "’").replace(":", "\\:").replace("%", "\\%")
        fc += (f";{last}drawtext=text='{t}':fontcolor=white:fontsize=58:borderw=5:"
               f"bordercolor=black:x=(w-text_w)/2:y=150:enable='lt(t,4)'[vt]")
        last = "[vt]"
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{m.start:.2f}", "-t", f"{m.end - m.start:.2f}",
         "-i", str(video), "-filter_complex", fc, "-map", last, "-map", "0:a?",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
         "-movflags", "+faststart", "-r", "30", str(out)])


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Find hype moments in a GTA 6 stream VOD and cut vertical clips.")
    ap.add_argument("source", help="VOD URL (Twitch/YouTube/Kick) or local video file")
    ap.add_argument("--chat", type=Path, help="chat replay file (JSON / JSON-lines / CSV) for a local video")
    ap.add_argument("--no-chat", action="store_true", help="don't download chat, use audio only")
    ap.add_argument("-o", "--out", type=Path, default=Path("clips"), help="output folder (default: clips)")
    ap.add_argument("--top", type=int, default=10, help="number of clips (default 10)")
    ap.add_argument("--pre", type=float, default=25, help="seconds before the peak (default 25)")
    ap.add_argument("--post", type=float, default=12, help="seconds after the peak (default 12)")
    ap.add_argument("--min-gap", type=float, default=90, help="min seconds between clips (default 90)")
    ap.add_argument("--chat-delay", type=float, default=6, help="chat reaction lag in seconds (default 6)")
    ap.add_argument("--audio-weight", type=float, default=0.45, help="0..1, audio vs chat weighting")
    ap.add_argument("--min-score", type=float, default=2.0, help="ignore peaks below this score (default 2)")
    ap.add_argument("--skip-start", type=float, default=0, help="ignore the first N seconds (intro/starting soon)")
    ap.add_argument("--layout", choices=["blur", "crop", "facecam"], default="blur",
                    help="blur = full frame on blurred bg; crop = center 9:16; facecam = cam on top, game below")
    ap.add_argument("--facecam", help="facecam box in source pixels: x,y,w,h (for --layout facecam)")
    ap.add_argument("--captions", action="store_true", help="burn in captions (needs faster-whisper)")
    ap.add_argument("--whisper-model", default="small", help="faster-whisper model (default small)")
    ap.add_argument("--title", help="text shown on the first 4s of every clip, e.g. streamer name")
    ap.add_argument("--max-height", type=int, default=1080, help="download quality cap (default 1080)")
    ap.add_argument("--crf", type=int, default=20, help="x264 quality, lower = better (default 20)")
    ap.add_argument("--dry-run", action="store_true", help="only list moments, don't render")
    args = ap.parse_args()

    need("ffmpeg"); need("ffprobe")
    facecam = None
    if args.layout == "facecam":
        if not args.facecam:
            sys.exit("error: --layout facecam needs --facecam x,y,w,h")
        facecam = tuple(int(v) for v in args.facecam.split(","))
        if len(facecam) != 4:
            sys.exit("error: --facecam must be x,y,w,h")

    args.out.mkdir(parents=True, exist_ok=True)
    chat_path = args.chat
    if re.match(r"https?://", args.source):
        video, dl_chat = download(args.source, args.out / "_source", args.max_height, not args.no_chat)
        chat_path = chat_path or dl_chat
    else:
        video = Path(args.source)
        if not video.exists():
            sys.exit(f"error: {video} not found")

    dur = duration_of(video)
    print(f"[info] {video.name}  duration {hms(dur)}")
    audio = audio_loudness(video, dur)
    if args.skip_start:
        audio[: int(args.skip_start)] = np.median(audio)

    chat = None
    if chat_path and not args.no_chat:
        msgs = load_chat(chat_path)
        print(f"[chat] {len(msgs)} messages loaded from {chat_path.name}")
        if msgs:
            chat = chat_signal(msgs, len(audio), args.chat_delay)
            if args.skip_start:
                chat[0][: int(args.skip_start)] = 0
                chat[1][: int(args.skip_start)] = 0

    moments = find_moments(audio, chat, args.top, args.pre, args.post, args.min_gap, dur,
                           args.audio_weight, args.min_score)
    if not moments:
        sys.exit("no hype moments found - try a lower --min-score or --audio-weight 1")

    print(f"\n{'#':>3}  {'time':>8}  {'clip':>19}  {'score':>6}  {'audio':>6}  {'chat':>6}  msgs")
    for m in moments:
        print(f"{m.rank:>3}  {hms(m.peak):>8}  {hms(m.start)}-{hms(m.end)}  "
              f"{m.score:6.2f}  {m.audio_z:6.2f}  {m.chat_z:6.2f}  {m.chat_msgs}")

    manifest = args.out / "moments.json"
    manifest.write_text(json.dumps([asdict(m) for m in moments], indent=2))
    with open(args.out / "moments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(moments[0]).keys()))
        w.writeheader()
        w.writerows(asdict(m) for m in moments)

    if args.dry_run:
        print(f"\n[dry-run] moments saved to {manifest}")
        return

    stem = re.sub(r"[^\w-]+", "_", video.stem)[:40]
    for m in moments:
        out = args.out / f"{m.rank:02d}_{stem}_{hms(m.peak).replace(':', '-')}.mp4"
        srt = transcribe(video, m, args.whisper_model, out.with_suffix(".srt")) if args.captions else None
        print(f"[render] {m.rank}/{len(moments)} -> {out.name}")
        render(video, m, out, args.layout, facecam, srt, args.title, args.crf)
    print(f"\ndone: {len(moments)} clips in {args.out.resolve()}")


if __name__ == "__main__":
    main()
