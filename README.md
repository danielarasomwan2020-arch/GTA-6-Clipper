# GTA 6 Streamer Clipper

Give it a stream VOD and it finds the hype moments from **audio spikes** (screaming, explosions, laughing) and **chat
explosions** (KEKW, "CLIP IT", ????, W spam). It then cuts them into **vertical 9:16 clips** for TikTok, Shorts and Reels.

## Setup
Needs Python 3.10+ and ffmpeg.
```bash
pip install -r requirements.txt
```
```bash
pip install faster-whisper   # only if you want burned-in captions
```

## Usage
```bash
python clipper.py "https://www.twitch.tv/videos/2290000000" --top 8 --captions --title "GTA 6 IS INSANE"
```
```bash
python clipper.py "https://www.youtube.com/watch?v=XXXX" --dry-run          # list moments only
```
```bash
python clipper.py vod.mp4 --chat chat.json --layout facecam --facecam 1500,40,380,215
```

Output goes to `clips/` as `01_<name>_<time>.mp4` and so on. You also get `moments.csv` and `moments.json` with timestamps and scores.

### Layouts
| `--layout` | Look |
|---|---|
| `blur` (default) | Full 16:9 gameplay in the middle with a blurred copy filling the frame |
| `crop` | Center 9:16 crop (good for close action) |
| `facecam` | Streamer cam on top (pass its box with `--facecam x,y,w,h`) and gameplay below. Best for reaction clips |

To get the facecam box, pause the VOD on a frame and read the cam's pixel position in any image editor at full resolution.

### Tuning
- `--pre 25 --post 12`: seconds kept before and after the peak. GTA moments need setup, so keep `--pre` long.
- `--skip-start 600`: ignore the "starting soon" screen.
- `--chat-delay 6`: chat reacts a few seconds after the moment. Raise it for streams with big delay.
- `--audio-weight 0.45`: lower it to trust chat more. Use 1.0 for audio only.
- `--min-score 2`, `--min-gap 90`, `--top 10`: control how picky the clipper is and how many clips you get.

### Chat input
Chat is fetched automatically for Twitch, YouTube and Kick URLs (via `chat-downloader`). For local files, `--chat` accepts:
- chat-downloader JSON
- TwitchDownloader JSON
- yt-dlp `live_chat.json`
- CSV with `time` and `message` columns

## Rules
Only clip streamers who allow it. Many big GTA RP and GTA 6 streamers have clipper programs or revenue splits, so check their
Discord. Credit the streamer in every caption.
