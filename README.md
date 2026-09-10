# YouTube Shorts Factory

A local video-analysis pipeline that watches a YouTube playlist, identifies
promising short-form moments, reframes them for portrait video, and presents a
review queue in a FastAPI dashboard.

The project combines scene detection, speech activity, optional transcription,
object/pose tracking, CLIP-based reranking, and FFmpeg rendering. Selection is
kept human-in-the-loop: generated clips are reviewed locally rather than being
uploaded automatically.

## Features

- playlist polling and durable SQLite-backed work queues;
- hybrid visual/transcript moment scoring;
- YOLO, MediaPipe, and scene-aware subject tracking;
- 1080 × 1920 rendering with optional word-level captions;
- browser dashboards for source videos, queued work, and generated shorts; and
- configurable model, duration, scoring, and rendering settings.

## Requirements

- Python 3.11+
- FFmpeg and FFprobe on `PATH`
- a YouTube Data API key and playlist ID
- substantial disk space; GPU acceleration is recommended for model-heavy runs

## Setup

```sh
python -m venv .venv
python -m pip install -r requirements.txt
cp .env.example .env
python main.py init
```

Add `YOUTUBE_API_KEY` and `YOUTUBE_PLAYLIST_ID` to `.env`. Model weights are
downloaded by their libraries when needed and are not stored in Git.

## Run

```sh
python main.py watch   # discover playlist videos
python main.py work    # analyse and render queued videos
python main.py web     # dashboard at http://localhost:8080
python main.py all     # watcher, worker, and dashboard together
```

Generated data lives under `data/` by default. See [`.env.example`](.env.example)
for the available settings.

## Architecture

The current prototype is intentionally contained in one executable module so
the pipeline can be run and profiled end to end. SQLite stores videos, analysis
attempts, candidate moments, and review state. The worker writes intermediate
and final media beneath the configured data directory, while FastAPI exposes
read-only dashboards plus explicit review toggles.

## Responsible use

Only download and transform videos you own or are authorised to reuse. YouTube
terms, copyright, performer consent, and music licensing still apply to derived
clips. API keys, downloaded source media, model weights, and rendered outputs
are excluded from version control.

## Verification

```sh
python -m py_compile main.py
```

Full end-to-end verification requires FFmpeg, model downloads, API access, and
a test playlist.
