from __future__ import annotations

import html
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

import cv2
import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from starlette.middleware.cors import CORSMiddleware

try:
    import webrtcvad
except Exception:
    webrtcvad = None


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.lstrip("\ufeff").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


ROOT = Path(__file__).resolve().parent
load_env_file(ROOT / ".env")


def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    youtube_playlist_id: str = env_str("YOUTUBE_PLAYLIST_ID", "")
    youtube_api_key: str = env_str("YOUTUBE_API_KEY", "")

    data_dir: Path = ROOT / env_str("DATA_DIR", "data")
    poll_interval_seconds: int = env_int("POLL_INTERVAL_SECONDS", 120)

    shorts_per_video: int = env_int("SHORTS_PER_VIDEO", 15)
    short_min_seconds: int = env_int("SHORT_MIN_SECONDS", 4)
    short_max_seconds: int = env_int("SHORT_MAX_SECONDS", 9)
    short_target_seconds: int = env_int("SHORT_TARGET_SECONDS", 7)

    moment_mode: str = env_str("MOMENT_MODE", "visual").lower()
    visual_sample_fps: float = env_float("VISUAL_SAMPLE_FPS", 2.0)
    visual_weight: float = env_float("VISUAL_WEIGHT", 0.95)
    transcript_weight: float = env_float("TRANSCRIPT_WEIGHT", 0.05)

    captions_mode: str = env_str("CAPTIONS_MODE", "auto").lower()
    speech_vad_aggressiveness: int = env_int("SPEECH_VAD_AGGRESSIVENESS", 2)
    speech_ratio_threshold: float = env_float("SPEECH_RATIO_THRESHOLD", 0.06)
    force_transcribe: bool = env_bool("FORCE_TRANSCRIBE", False)

    auto_select_top_n: int = env_int("AUTO_SELECT_TOP_N", 3)

    render_width: int = env_int("RENDER_WIDTH", 1080)
    render_height: int = env_int("RENDER_HEIGHT", 1920)

    scene_content_threshold: float = env_float("SCENE_CONTENT_THRESHOLD", 32.0)
    scene_adaptive_threshold: float = env_float("SCENE_ADAPTIVE_THRESHOLD", 3.0)
    scene_min_seconds: float = env_float("SCENE_MIN_SECONDS", 1.0)

    yolo_model: str = env_str("YOLO_MODEL", "yolov8n.pt")
    yolo_conf: float = env_float("YOLO_CONF", 0.20)
    yolo_iou: float = env_float("YOLO_IOU", 0.45)
    yolo_tracker: str = env_str("YOLO_TRACKER", "bytetrack.yaml")

    pose_min_detection_confidence: float = env_float("POSE_MIN_DETECTION_CONFIDENCE", 0.50)
    pose_min_tracking_confidence: float = env_float("POSE_MIN_TRACKING_CONFIDENCE", 0.50)

    enable_clip_rerank: bool = env_bool("ENABLE_CLIP_RERANK", True)
    clip_model_name: str = env_str("CLIP_MODEL_NAME", "ViT-B-32")
    clip_pretrained: str = env_str("CLIP_PRETRAINED", "laion2b_s34b_b79k")
    clip_frame_samples: int = env_int("CLIP_FRAME_SAMPLES", 4)

    visual_analyse_all_frames: bool = env_bool("VISUAL_ANALYSE_ALL_FRAMES", True)
    visual_progress_update_every: int = env_int("VISUAL_PROGRESS_UPDATE_EVERY", 120)

    attempt_start_persist_seconds: float = env_float("ATTEMPT_START_PERSIST_SECONDS", 0.9)
    attempt_min_seconds: float = env_float("ATTEMPT_MIN_SECONDS", 2.5)
    attempt_max_gap_seconds: float = env_float("ATTEMPT_MAX_GAP_SECONDS", 0.45)
    attempt_top_stationary_seconds: float = env_float("ATTEMPT_TOP_STATIONARY_SECONDS", 0.75)
    attempt_drop_velocity_threshold: float = env_float("ATTEMPT_DROP_VELOCITY_THRESHOLD", 0.35)
    attempt_min_vertical_progress: float = env_float("ATTEMPT_MIN_VERTICAL_PROGRESS", 0.10)
    attempt_top_margin: float = env_float("ATTEMPT_TOP_MARGIN", 0.045)

    caption_max_words: int = env_int("CAPTION_MAX_WORDS", 4)
    caption_max_duration: float = env_float("CAPTION_MAX_DURATION", 1.2)
    caption_max_gap: float = env_float("CAPTION_MAX_GAP", 0.22)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def shorts_dir(self) -> Path:
        return self.data_dir / "shorts"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.shorts_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
_MODEL_CACHE: dict[str, Any] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def db() -> sqlite3.Connection:
    SETTINGS.ensure_dirs()
    conn = sqlite3.connect(SETTINGS.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY,
                playlist_id TEXT NOT NULL,
                title TEXT NOT NULL,
                channel_title TEXT NOT NULL,
                added_at TEXT NOT NULL,
                status TEXT NOT NULL,
                raw_path TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shorts (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                start_sec REAL NOT NULL,
                end_sec REAL NOT NULL,
                score REAL NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                short_path TEXT NOT NULL,
                thumb_path TEXT NOT NULL,
                transcript TEXT NOT NULL DEFAULT '',
                uploaded_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL,
                message TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                level TEXT NOT NULL,
                source TEXT NOT NULL,
                message TEXT NOT NULL,
                job_id TEXT NOT NULL DEFAULT '',
                video_id TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.commit()


def log_event(level: str, source: str, message: str, job_id: str = "", video_id: str = "") -> None:
    with db() as conn:
        conn.execute(
            """
            INSERT INTO app_logs (created_at, level, source, message, job_id, video_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (utc_now(), level.upper(), source, message[:1200], job_id, video_id),
        )
        conn.commit()


def recover_interrupted_jobs() -> None:
    with db() as conn:
        rows = conn.execute(
            "SELECT id, type, payload_json FROM jobs WHERE status = 'running'"
        ).fetchall()

        for row in rows:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    progress = 0,
                    message = 'Recovered after restart',
                    updated_at = ?
                WHERE id = ?
                """,
                (utc_now(), row["id"]),
            )

        conn.commit()

    for row in rows:
        video_id = ""
        try:
            payload = json.loads(row["payload_json"])
            video_id = payload.get("video_id", "") or payload.get("short_id", "")
        except Exception:
            pass
        log_event(
            "WARNING",
            "worker",
            f"Recovered interrupted {row['type']} job after restart.",
            job_id=row["id"],
            video_id=video_id,
        )


def enqueue_job(job_type: str, payload: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex
    now = utc_now()
    with db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, type, status, progress, message, payload_json, created_at, updated_at)
            VALUES (?, ?, 'queued', 0, 'Queued', ?, ?, ?)
            """,
            (job_id, job_type, json.dumps(payload), now, now),
        )
        conn.commit()
    return job_id


def update_job(job_id: str, *, status: str | None = None, progress: int | None = None, message: str | None = None) -> None:
    fields: list[str] = []
    params: list[Any] = []

    if status is not None:
        fields.append("status = ?")
        params.append(status)

    if progress is not None:
        fields.append("progress = ?")
        params.append(progress)

    if message is not None:
        fields.append("message = ?")
        params.append(message[:400])

    fields.append("updated_at = ?")
    params.append(utc_now())
    params.append(job_id)

    with db() as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", params)
        conn.commit()


def next_queued_job() -> sqlite3.Row | None:
    with db() as conn:
        return conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()


def safe_rel(path: Path) -> str:
    return str(path.resolve().relative_to(SETTINGS.data_dir.resolve()))


def resolve_media_path(rel_or_abs: str) -> Path:
    candidate = Path(rel_or_abs)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (SETTINGS.data_dir / candidate).resolve()

    root = SETTINGS.data_dir.resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError("Path is outside data directory.")

    return resolved


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]


@dataclass(frozen=True)
class ClipPlan:
    start: float
    end: float
    title_hint: str
    visual_score: float
    attempt_id: int = -1
    kind: str = "highlight"


@dataclass(frozen=True)
class SceneSpan:
    index: int
    start: float
    end: float


@dataclass
class FrameSample:
    frame_idx: int
    t: float
    scene_idx: int
    track_id: int
    bbox: tuple[float, float, float, float]
    center_x: float
    center_y: float
    area: float
    torso: tuple[float, float]
    points: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class Attempt:
    attempt_id: int
    scene_idx: int
    start_i: int
    end_i: int
    peak_i: int
    finish_i: int
    finish_kind: str
    completion: float


def get_video_metadata(video_path: Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0
    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration,
    }


def smooth_series(values: np.ndarray, window_frames: int) -> np.ndarray:
    window_frames = max(1, int(window_frames))
    if len(values) == 0 or window_frames <= 1:
        return values
    kernel = np.ones(window_frames, dtype=np.float32) / float(window_frames)
    return np.convolve(values, kernel, mode="same")


def normalize_by_percentile(values: np.ndarray, percentile: float = 95.0, ceiling: float = 2.0) -> np.ndarray:
    if len(values) == 0:
        return values
    scale = float(np.percentile(values, percentile)) + 1e-6
    return np.clip(values / scale, 0.0, ceiling)


def point_distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def mean_point(points: list[tuple[float, float] | None]) -> tuple[float, float] | None:
    valid = [p for p in points if p is not None]
    if not valid:
        return None
    return (
        float(sum(p[0] for p in valid) / len(valid)),
        float(sum(p[1] for p in valid) / len(valid)),
    )


def overlaps(a: ClipPlan, b: ClipPlan) -> bool:
    return max(a.start, b.start) < min(a.end, b.end)


def dedupe_clip_plans(plans: list[ClipPlan]) -> list[ClipPlan]:
    chosen: list[ClipPlan] = []
    for plan in plans:
        if any(overlaps(plan, existing) for existing in chosen):
            continue
        chosen.append(plan)
    return chosen


def segments_in_window(segments: list[Segment], start: float, end: float) -> list[Segment]:
    return [seg for seg in segments if not (seg.end <= start or seg.start >= end)]


def segment_to_dict(segment: Segment) -> dict[str, Any]:
    return {
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "words": [{"start": w.start, "end": w.end, "text": w.text} for w in segment.words],
    }


def extract_window_text(segments: list[Segment], start: float, end: float) -> str:
    return " ".join(seg.text for seg in segments_in_window(segments, start, end)).strip()


def segment_window_score(text: str, duration: float) -> float:
    lower = text.lower()
    hook_words = ["why", "how", "what", "crazy", "wait", "watch", "listen", "don't", "you", "no way", "come on"]
    hook_score = sum(0.10 for word in hook_words if word in lower)
    punctuation_score = 0.12 if "?" in text else 0.0
    punctuation_score += 0.08 if "!" in text else 0.0
    words = len(text.split())
    density_score = min(1.0, words / max(1.0, duration * 2.5))
    length_score = min(0.45, words / 55.0)
    return hook_score + punctuation_score + density_score + length_score


def compute_text_score(text: str, duration: float) -> float:
    if not text:
        return 0.0
    score = segment_window_score(text, duration)
    return max(0.0, min(10.0, score * 1.6))


def transcript_clip_plans(segments: list[Segment]) -> list[ClipPlan]:
    plans: list[ClipPlan] = []
    if not segments:
        return plans

    for i in range(len(segments)):
        start = segments[i].start
        end = start
        text_parts: list[str] = []

        for j in range(i, len(segments)):
            end = segments[j].end
            text_parts.append(segments[j].text)
            if (end - start) >= SETTINGS.short_target_seconds:
                break

        duration = end - start
        if duration < SETTINGS.short_min_seconds or duration > SETTINGS.short_max_seconds:
            continue

        text = " ".join(text_parts).strip()
        score = compute_text_score(text, duration)
        plans.append(
            ClipPlan(
                start=round(start, 3),
                end=round(end, 3),
                title_hint=(text[:90] + "...") if len(text) > 90 else text,
                visual_score=score,
                attempt_id=-1,
                kind="transcript",
            )
        )

    plans.sort(key=lambda plan: plan.visual_score, reverse=True)
    return dedupe_clip_plans(plans)


def list_playlist_items() -> list[dict[str, str]]:
    if not SETTINGS.youtube_playlist_id or not SETTINGS.youtube_api_key:
        raise RuntimeError("YOUTUBE_PLAYLIST_ID and YOUTUBE_API_KEY are required.")

    items: list[dict[str, str]] = []
    page_token = ""

    while True:
        params = {
            "part": "snippet,contentDetails",
            "playlistId": SETTINGS.youtube_playlist_id,
            "maxResults": 50,
            "key": SETTINGS.youtube_api_key,
        }
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(
            "https://www.googleapis.com/youtube/v3/playlistItems",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            content_details = item.get("contentDetails", {})
            video_id = content_details.get("videoId")
            if not video_id:
                continue

            items.append(
                {
                    "video_id": video_id,
                    "title": snippet.get("title", "") or video_id,
                    "channel_title": snippet.get("videoOwnerChannelTitle") or snippet.get("channelTitle", ""),
                    "added_at": snippet.get("publishedAt", utc_now()),
                }
            )

        page_token = data.get("nextPageToken", "")
        if not page_token:
            break

    items.sort(key=lambda item: item["added_at"])
    return items


def download_video(video_id: str) -> Path:
    SETTINGS.ensure_dirs()
    output_template = str(SETTINGS.raw_dir / f"{video_id}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            "bestvideo+bestaudio/best",
            "--merge-output-format",
            "mp4",
            "-o",
            output_template,
            url,
        ],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "yt-dlp failed")

    mp4 = SETTINGS.raw_dir / f"{video_id}.mp4"
    if mp4.exists():
        return mp4

    for file in SETTINGS.raw_dir.glob(f"{video_id}.*"):
        if file.is_file():
            return file

    raise RuntimeError("Downloaded file not found.")


def speech_ratio_from_video(video_path: Path, frame_ms: int = 30) -> float:
    if webrtcvad is None:
        return 1.0

    vad = webrtcvad.Vad(max(0, min(3, SETTINGS.speech_vad_aggressiveness)))

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                str(wav_path),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return 1.0

        with wave.open(str(wav_path), "rb") as wf:
            pcm = wf.readframes(wf.getnframes())

    frame_bytes = int(16000 * (frame_ms / 1000.0) * 2)
    total = 0
    speech = 0
    offset = 0

    while offset + frame_bytes <= len(pcm):
        frame = pcm[offset : offset + frame_bytes]
        offset += frame_bytes
        total += 1
        if vad.is_speech(frame, 16000):
            speech += 1

    return float(speech) / float(max(1, total))


def should_use_transcript(speech_ratio: float) -> bool:
    if SETTINGS.force_transcribe:
        return True
    return speech_ratio >= SETTINGS.speech_ratio_threshold


def captions_enabled(speech_ratio: float) -> bool:
    if SETTINGS.captions_mode == "on":
        return True
    if SETTINGS.captions_mode == "off":
        return False
    return speech_ratio >= SETTINGS.speech_ratio_threshold


def transcribe_video(video_path: Path) -> list[Segment]:
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return []

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(video_path),
        vad_filter=True,
        word_timestamps=True,
        beam_size=5,
    )

    out: list[Segment] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue

        words: list[Word] = []
        for word in getattr(seg, "words", []) or []:
            if word.start is None or word.end is None:
                continue
            token = (word.word or "").strip()
            if not token:
                continue
            words.append(
                Word(
                    start=float(word.start),
                    end=float(word.end),
                    text=token,
                )
            )

        out.append(
            Segment(
                start=float(seg.start),
                end=float(seg.end),
                text=text,
                words=words,
            )
        )

    return out


def get_yolo_model():
    if "yolo_model" not in _MODEL_CACHE:
        from ultralytics import YOLO

        _MODEL_CACHE["yolo_model"] = YOLO(SETTINGS.yolo_model)
    return _MODEL_CACHE["yolo_model"]


def get_pose_engine():
    if "pose_engine" not in _MODEL_CACHE:
        import mediapipe as mp

        _MODEL_CACHE["pose_mp"] = mp
        _MODEL_CACHE["pose_engine"] = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=SETTINGS.pose_min_detection_confidence,
            min_tracking_confidence=SETTINGS.pose_min_tracking_confidence,
        )
    return _MODEL_CACHE["pose_engine"]


def get_clip_bundle():
    if not SETTINGS.enable_clip_rerank:
        return None

    if "clip_bundle" in _MODEL_CACHE:
        return _MODEL_CACHE["clip_bundle"]

    try:
        import torch
        import open_clip
    except Exception:
        _MODEL_CACHE["clip_bundle"] = None
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        SETTINGS.clip_model_name,
        pretrained=SETTINGS.clip_pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(SETTINGS.clip_model_name)

    prompts = [
        "a climber starting a climb on a board",
        "a climber doing a hard move on a climbing board",
        "a climber reaching the top hold on a climbing board",
        "a climber falling from a climbing board",
    ]

    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    bundle = {
        "torch": torch,
        "model": model,
        "preprocess": preprocess,
        "device": device,
        "prompts": prompts,
        "text_features": text_features,
    }
    _MODEL_CACHE["clip_bundle"] = bundle
    return bundle


def detect_scene_spans(video_path: Path) -> list[SceneSpan]:
    meta = get_video_metadata(video_path)
    total_duration = float(meta["duration"])

    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import AdaptiveDetector, ContentDetector
    except Exception:
        return [SceneSpan(index=0, start=0.0, end=total_duration)]

    video = open_video(str(video_path))
    manager = SceneManager()
    min_scene_len_frames = max(1, int(round(SETTINGS.scene_min_seconds * max(1.0, meta["fps"]))))

    manager.add_detector(
        AdaptiveDetector(
            adaptive_threshold=SETTINGS.scene_adaptive_threshold,
            min_scene_len=min_scene_len_frames,
        )
    )
    manager.add_detector(
        ContentDetector(
            threshold=SETTINGS.scene_content_threshold,
            min_scene_len=min_scene_len_frames,
        )
    )

    try:
        manager.detect_scenes(video=video, show_progress=False)
        scene_list = manager.get_scene_list()
    except Exception:
        return [SceneSpan(index=0, start=0.0, end=total_duration)]

    if not scene_list:
        return [SceneSpan(index=0, start=0.0, end=total_duration)]

    spans: list[SceneSpan] = []
    for i, (start_tc, end_tc) in enumerate(scene_list):
        spans.append(SceneSpan(index=i, start=float(start_tc.get_seconds()), end=float(end_tc.get_seconds())))
    return spans


def pick_tracked_person(boxes, preferred_track_id: int | None) -> tuple[int, int, int, int, int] | None:
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy), dtype=int)

    candidates: list[tuple[int, int, int, int, int, float]] = []
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = float(w * h)
        if area <= 0:
            continue
        candidates.append((int(ids[i]), x1, y1, x2, y2, area))

    if not candidates:
        return None

    if preferred_track_id is not None:
        for candidate in candidates:
            if candidate[0] == preferred_track_id:
                return candidate[:5]

    candidates.sort(key=lambda item: item[5], reverse=True)
    return candidates[0][:5]


def extract_pose_points(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> dict[str, tuple[float, float]]:
    pose = get_pose_engine()
    mp = _MODEL_CACHE["pose_mp"]

    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return {}

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if not getattr(result, "pose_landmarks", None):
        return {}

    landmarks = result.pose_landmarks.landmark
    crop_h, crop_w = crop.shape[:2]

    def global_point(enum_value) -> tuple[float, float] | None:
        landmark = landmarks[enum_value]
        if getattr(landmark, "visibility", 1.0) < 0.35:
            return None
        px = (x1 + (landmark.x * crop_w)) / max(1, frame_w)
        py = (y1 + (landmark.y * crop_h)) / max(1, frame_h)
        return (float(px), float(py))

    L = mp.solutions.pose.PoseLandmark
    points = {
        "left_shoulder": global_point(L.LEFT_SHOULDER.value),
        "right_shoulder": global_point(L.RIGHT_SHOULDER.value),
        "left_hip": global_point(L.LEFT_HIP.value),
        "right_hip": global_point(L.RIGHT_HIP.value),
        "left_wrist": global_point(L.LEFT_WRIST.value),
        "right_wrist": global_point(L.RIGHT_WRIST.value),
        "left_ankle": global_point(L.LEFT_ANKLE.value),
        "right_ankle": global_point(L.RIGHT_ANKLE.value),
        "nose": global_point(L.NOSE.value),
    }

    return {key: value for key, value in points.items() if value is not None}


def run_tracking_pose(
    video_path: Path,
    scene_spans: list[SceneSpan],
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[list[FrameSample], dict[str, float]]:
    meta = get_video_metadata(video_path)
    fps = max(1.0, float(meta["fps"]))
    total_frames = int(meta["total_frames"])
    frame_w = int(meta["width"])
    frame_h = int(meta["height"])

    model = get_yolo_model()

    scene_cursor = 0
    preferred_track_id: int | None = None
    next_progress = max(1, SETTINGS.visual_progress_update_every)

    samples: list[FrameSample] = []

    results_stream = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        tracker=SETTINGS.yolo_tracker,
        classes=[0],
        conf=SETTINGS.yolo_conf,
        iou=SETTINGS.yolo_iou,
        verbose=False,
    )

    for frame_idx, result in enumerate(results_stream, start=1):
        t = float((frame_idx - 1) / fps)

        while scene_cursor + 1 < len(scene_spans) and t >= scene_spans[scene_cursor].end:
            scene_cursor += 1
            preferred_track_id = None

        if progress_cb is not None and (frame_idx >= next_progress or frame_idx == total_frames):
            progress_cb(frame_idx, total_frames)
            next_progress = frame_idx + max(1, SETTINGS.visual_progress_update_every)

        if getattr(result, "orig_img", None) is None:
            continue

        picked = pick_tracked_person(getattr(result, "boxes", None), preferred_track_id)
        if picked is None:
            continue

        track_id, x1, y1, x2, y2 = picked
        preferred_track_id = track_id

        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(x1 + 1, min(frame_w, x2))
        y2 = max(y1 + 1, min(frame_h, y2))

        points = extract_pose_points(result.orig_img, (x1, y1, x2, y2), frame_w, frame_h)

        center_x = float(((x1 + x2) / 2.0) / max(1, frame_w))
        center_y = float(((y1 + y2) / 2.0) / max(1, frame_h))
        area = float(((x2 - x1) * (y2 - y1)) / max(1, frame_w * frame_h))

        torso = mean_point(
            [
                points.get("left_shoulder"),
                points.get("right_shoulder"),
                points.get("left_hip"),
                points.get("right_hip"),
            ]
        )
        if torso is None:
            torso = (center_x, center_y)

        samples.append(
            FrameSample(
                frame_idx=frame_idx,
                t=t,
                scene_idx=scene_spans[scene_cursor].index if scene_spans else 0,
                track_id=track_id,
                bbox=(x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h),
                center_x=center_x,
                center_y=center_y,
                area=area,
                torso=torso,
                points=points,
            )
        )

    if progress_cb is not None and total_frames > 0:
        progress_cb(total_frames, total_frames)

    return samples, meta


def detect_attempts(scene_samples: list[FrameSample], fps: float) -> tuple[list[Attempt], dict[str, np.ndarray]]:
    if len(scene_samples) < 8:
        return [], {}

    times = np.array([sample.t for sample in scene_samples], dtype=np.float32)
    center_x = np.array([sample.center_x for sample in scene_samples], dtype=np.float32)
    torso_y = np.array([sample.torso[1] for sample in scene_samples], dtype=np.float32)
    area = np.array([sample.area for sample in scene_samples], dtype=np.float32)
    pose_valid = np.array([1.0 if len(sample.points) >= 4 else 0.0 for sample in scene_samples], dtype=np.float32)

    dt = np.diff(times, prepend=times[0])
    dt[dt <= 0] = 1.0 / max(1.0, fps)

    upward = np.zeros_like(times)
    downward = np.zeros_like(times)
    lateral = np.zeros_like(times)
    limb_velocity = np.zeros_like(times)
    pose_change = np.zeros_like(times)

    for i in range(1, len(scene_samples)):
        upward[i] = max(0.0, float(torso_y[i - 1] - torso_y[i])) / float(dt[i])
        downward[i] = max(0.0, float(torso_y[i] - torso_y[i - 1])) / float(dt[i])
        lateral[i] = abs(float(center_x[i] - center_x[i - 1])) / float(dt[i])

        limb_names = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
        limb_deltas: list[float] = []
        point_deltas: list[float] = []

        for name in limb_names:
            if name in scene_samples[i - 1].points and name in scene_samples[i].points:
                limb_deltas.append(
                    point_distance(scene_samples[i - 1].points[name], scene_samples[i].points[name]) / float(dt[i])
                )

        common_keys = set(scene_samples[i - 1].points).intersection(scene_samples[i].points)
        for name in common_keys:
            point_deltas.append(
                point_distance(scene_samples[i - 1].points[name], scene_samples[i].points[name]) / float(dt[i])
            )

        if limb_deltas:
            limb_velocity[i] = float(np.mean(limb_deltas))
        if point_deltas:
            pose_change[i] = float(np.mean(point_deltas))

    upward_n = normalize_by_percentile(upward)
    downward_n = normalize_by_percentile(downward)
    lateral_n = normalize_by_percentile(lateral)
    limb_n = normalize_by_percentile(limb_velocity)
    pose_n = normalize_by_percentile(pose_change)

    effective_fps = max(1.0, fps)
    burst = smooth_series(
        (0.38 * upward_n) + (0.18 * lateral_n) + (0.22 * limb_n) + (0.22 * pose_n),
        int(round(effective_fps * 0.35)),
    )
    sustained = smooth_series(
        (0.32 * upward_n) + (0.16 * lateral_n) + (0.26 * limb_n) + (0.26 * pose_n),
        int(round(effective_fps * 0.85)),
    )

    x_median = float(np.median(center_x))
    x_iqr = float(np.percentile(center_x, 75) - np.percentile(center_x, 25))
    area_floor = float(np.percentile(area, 10)) * 0.55
    near_board = (
        (np.abs(center_x - x_median) <= max(0.08, x_iqr * 2.0))
        & (area >= max(0.005, area_floor))
        & (pose_valid > 0.0)
    )

    activity_threshold = max(0.16, float(np.percentile(sustained, 70)) * 0.90)
    low_threshold = activity_threshold * 0.52

    start_frames = max(1, int(round(SETTINGS.attempt_start_persist_seconds * effective_fps)))
    gap_frames = max(1, int(round(SETTINGS.attempt_max_gap_seconds * effective_fps)))
    top_stationary_frames = max(1, int(round(SETTINGS.attempt_top_stationary_seconds * effective_fps)))
    min_attempt_frames = max(1, int(round(SETTINGS.attempt_min_seconds * effective_fps)))

    attempts: list[Attempt] = []

    in_attempt = False
    start_streak = 0
    inactivity_streak = 0
    stationary_top_streak = 0
    attempt_start_i = 0
    attempt_min_torso = 0.0
    attempt_top_i = 0

    for i in range(len(scene_samples)):
        active_now = bool(near_board[i] and (sustained[i] >= activity_threshold or burst[i] >= activity_threshold * 1.15))

        if not in_attempt:
            if active_now:
                start_streak += 1
                if start_streak >= start_frames:
                    in_attempt = True
                    attempt_start_i = i - start_streak + 1
                    inactivity_streak = 0
                    stationary_top_streak = 0
                    attempt_min_torso = float(torso_y[attempt_start_i])
                    attempt_top_i = attempt_start_i
            else:
                start_streak = 0
            continue

        if torso_y[i] < attempt_min_torso:
            attempt_min_torso = float(torso_y[i])
            attempt_top_i = i

        vertical_progress = max(0.0, float(torso_y[attempt_start_i] - attempt_min_torso))
        top_like = (
            vertical_progress >= SETTINGS.attempt_min_vertical_progress
            and torso_y[i] <= (attempt_min_torso + SETTINGS.attempt_top_margin)
            and sustained[i] <= low_threshold
        )

        if top_like:
            stationary_top_streak += 1
        else:
            stationary_top_streak = 0

        if downward_n[i] >= SETTINGS.attempt_drop_velocity_threshold and vertical_progress >= (SETTINGS.attempt_min_vertical_progress * 0.60):
            attempt_end_i = i
            if (attempt_end_i - attempt_start_i + 1) >= min_attempt_frames:
                completion = min(0.85, 0.55 + (vertical_progress * 2.2))
                attempts.append(
                    Attempt(
                        attempt_id=len(attempts),
                        scene_idx=scene_samples[attempt_start_i].scene_idx,
                        start_i=attempt_start_i,
                        end_i=attempt_end_i,
                        peak_i=attempt_top_i,
                        finish_i=i,
                        finish_kind="fall",
                        completion=float(completion),
                    )
                )
            in_attempt = False
            start_streak = 0
            inactivity_streak = 0
            stationary_top_streak = 0
            continue

        if stationary_top_streak >= top_stationary_frames:
            attempt_end_i = i
            if (attempt_end_i - attempt_start_i + 1) >= min_attempt_frames:
                attempts.append(
                    Attempt(
                        attempt_id=len(attempts),
                        scene_idx=scene_samples[attempt_start_i].scene_idx,
                        start_i=attempt_start_i,
                        end_i=attempt_end_i,
                        peak_i=attempt_top_i,
                        finish_i=i,
                        finish_kind="top",
                        completion=1.0,
                    )
                )
            in_attempt = False
            start_streak = 0
            inactivity_streak = 0
            stationary_top_streak = 0
            continue

        if not near_board[i] or sustained[i] <= low_threshold:
            inactivity_streak += 1
            if inactivity_streak >= gap_frames:
                attempt_end_i = max(attempt_start_i, i - inactivity_streak)
                if (attempt_end_i - attempt_start_i + 1) >= min_attempt_frames:
                    completion = min(0.65, max(0.10, vertical_progress / max(0.12, SETTINGS.attempt_min_vertical_progress * 2.0)))
                    attempts.append(
                        Attempt(
                            attempt_id=len(attempts),
                            scene_idx=scene_samples[attempt_start_i].scene_idx,
                            start_i=attempt_start_i,
                            end_i=attempt_end_i,
                            peak_i=attempt_top_i,
                            finish_i=attempt_top_i,
                            finish_kind="cut",
                            completion=float(completion),
                        )
                    )
                in_attempt = False
                start_streak = 0
                inactivity_streak = 0
                stationary_top_streak = 0
        else:
            inactivity_streak = 0

    if in_attempt:
        attempt_end_i = len(scene_samples) - 1
        vertical_progress = max(0.0, float(torso_y[attempt_start_i] - attempt_min_torso))
        if (attempt_end_i - attempt_start_i + 1) >= min_attempt_frames:
            completion = min(0.65, max(0.10, vertical_progress / max(0.12, SETTINGS.attempt_min_vertical_progress * 2.0)))
            attempts.append(
                Attempt(
                    attempt_id=len(attempts),
                    scene_idx=scene_samples[attempt_start_i].scene_idx,
                    start_i=attempt_start_i,
                    end_i=attempt_end_i,
                    peak_i=attempt_top_i,
                    finish_i=attempt_top_i,
                    finish_kind="cut",
                    completion=float(completion),
                )
            )

    metrics = {
        "times": times,
        "center_x": center_x,
        "torso_y": torso_y,
        "upward_n": upward_n,
        "downward_n": downward_n,
        "lateral_n": lateral_n,
        "limb_n": limb_n,
        "pose_n": pose_n,
        "burst": burst,
        "sustained": sustained,
        "near_board": near_board.astype(np.float32),
        "active": ((near_board) & (sustained >= activity_threshold)).astype(np.float32),
    }
    return attempts, metrics


def sample_frames_for_window(video_path: Path, start: float, end: float, sample_count: int) -> list[np.ndarray]:
    sample_count = max(1, sample_count)
    if end <= start:
        return []

    times = np.linspace(start, end, num=sample_count, dtype=np.float32)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames: list[np.ndarray] = []
    for ts in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts * 1000.0))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)

    cap.release()
    return frames


def score_clip_prompt_similarity(video_path: Path, start: float, end: float, finish_kind: str) -> float:
    bundle = get_clip_bundle()
    if bundle is None:
        return 0.0

    frames = sample_frames_for_window(video_path, start, end, SETTINGS.clip_frame_samples)
    if not frames:
        return 0.0

    torch = bundle["torch"]
    model = bundle["model"]
    preprocess = bundle["preprocess"]
    device = bundle["device"]
    text_features = bundle["text_features"]

    images = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(preprocess(Image.fromarray(rgb)))

    image_tensor = torch.stack(images).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sims = image_features @ text_features.T

    prompt_scores = sims.max(dim=0).values.detach().cpu().numpy()
    prompt_scores = (prompt_scores + 1.0) / 2.0

    if finish_kind == "top":
        weights = np.array([0.10, 0.30, 0.50, 0.10], dtype=np.float32)
    elif finish_kind == "fall":
        weights = np.array([0.05, 0.30, 0.10, 0.55], dtype=np.float32)
    else:
        weights = np.array([0.10, 0.55, 0.20, 0.15], dtype=np.float32)

    return float(np.clip(np.sum(prompt_scores * weights), 0.0, 1.0))


def score_attempt_candidates(
    video_path: Path,
    scene_samples: list[FrameSample],
    attempts: list[Attempt],
    metrics: dict[str, np.ndarray],
    segments: list[Segment],
) -> list[ClipPlan]:
    if not attempts:
        return []

    times = metrics["times"]
    torso_y = metrics["torso_y"]
    pose_n = metrics["pose_n"]
    limb_n = metrics["limb_n"]
    upward_n = metrics["upward_n"]
    burst = metrics["burst"]
    active = metrics["active"]

    primary_candidates: list[tuple[ClipPlan, float]] = []
    secondary_candidates: list[tuple[ClipPlan, float]] = []

    for attempt in attempts:
        attempt_times = times[attempt.start_i:attempt.end_i + 1]
        if len(attempt_times) < 4:
            continue

        attempt_start = float(attempt_times[0])
        attempt_end = float(attempt_times[-1])
        attempt_duration = max(0.01, attempt_end - attempt_start)
        finish_time = float(times[attempt.finish_i])
        peak_time = float(times[attempt.peak_i])

        local_pose = pose_n[attempt.start_i:attempt.end_i + 1]
        local_limb = limb_n[attempt.start_i:attempt.end_i + 1]
        local_upward = upward_n[attempt.start_i:attempt.end_i + 1]
        local_burst = burst[attempt.start_i:attempt.end_i + 1]
        local_active = active[attempt.start_i:attempt.end_i + 1]
        local_torso = torso_y[attempt.start_i:attempt.end_i + 1]

        action_curve = (0.42 * local_pose) + (0.23 * local_limb) + (0.20 * local_upward) + (0.15 * local_burst)
        crux_local_i = int(np.argmax(action_curve))
        crux_global_i = attempt.start_i + crux_local_i
        crux_time = float(times[crux_global_i])

        if attempt.finish_kind in {"top", "fall"}:
            anchor_time = finish_time
            primary_start = max(attempt_start - 0.35, anchor_time - (SETTINGS.short_target_seconds - 0.7))
            primary_end = min(attempt_end + 0.45, primary_start + SETTINGS.short_target_seconds)
            if primary_end < anchor_time + 0.35:
                primary_end = min(attempt_end + 0.55, anchor_time + 0.35)
                primary_start = max(attempt_start - 0.35, primary_end - SETTINGS.short_target_seconds)
        else:
            anchor_time = crux_time
            primary_start = max(attempt_start - 0.25, anchor_time - (SETTINGS.short_target_seconds * 0.55))
            primary_end = min(attempt_end + 0.30, primary_start + SETTINGS.short_target_seconds)

        if (primary_end - primary_start) < SETTINGS.short_min_seconds:
            primary_end = min(attempt_end + 0.45, primary_start + SETTINGS.short_min_seconds)

        candidate_windows: list[tuple[float, float, str]] = [
            (
                round(float(primary_start), 3),
                round(float(min(primary_end, primary_start + SETTINGS.short_max_seconds)), 3),
                "primary",
            )
        ]

        if attempt_duration >= 11.0 and abs(crux_time - finish_time) >= 2.0:
            secondary_start = max(attempt_start - 0.15, crux_time - (SETTINGS.short_target_seconds * 0.45))
            secondary_end = min(attempt_end + 0.20, secondary_start + SETTINGS.short_target_seconds)
            if (secondary_end - secondary_start) >= SETTINGS.short_min_seconds:
                candidate_windows.append(
                    (
                        round(float(secondary_start), 3),
                        round(float(min(secondary_end, secondary_start + SETTINGS.short_max_seconds)), 3),
                        "secondary",
                    )
                )

        scored: list[tuple[ClipPlan, float]] = []
        total_progress = max(1e-4, float(torso_y[attempt.start_i] - np.min(local_torso)))

        for window_start, window_end, kind in candidate_windows:
            mask = (times >= window_start) & (times <= window_end)
            if not np.any(mask):
                continue

            window_pose = pose_n[mask]
            window_torso = torso_y[mask]
            window_active = active[mask]

            peak_pose_change = float(np.max(window_pose)) if len(window_pose) else 0.0
            vertical_progress = max(0.0, float(window_torso[0] - np.min(window_torso))) / total_progress
            vertical_progress = float(np.clip(vertical_progress, 0.0, 1.0))
            action_density = float(np.mean(window_active)) if len(window_active) else 0.0

            finish_event = 1.0 if (attempt.finish_kind in {"top", "fall"} and window_start <= finish_time <= window_end) else 0.25
            transcript_excitement = compute_text_score(
                extract_window_text(segments, window_start, window_end),
                window_end - window_start,
            ) / 10.0
            clip_prompt_similarity = score_clip_prompt_similarity(video_path, window_start, window_end, attempt.finish_kind)

            score = (
                (0.35 * attempt.completion)
                + (0.20 * peak_pose_change)
                + (0.15 * vertical_progress)
                + (0.15 * action_density)
                + (0.10 * finish_event)
                + (0.05 * transcript_excitement)
            )

            score += (0.05 * clip_prompt_similarity)

            if window_start > (attempt_start + 2.0):
                score -= 0.08

            score = float(np.clip(score, 0.0, 1.0))
            score_10 = round(score * 10.0, 2)

            hint = (
                f"attempt {attempt.attempt_id + 1} {attempt.finish_kind}; "
                f"completion={attempt.completion:.2f}; "
                f"pose={peak_pose_change:.2f}; "
                f"progress={vertical_progress:.2f}; "
                f"density={action_density:.2f}; "
                f"clip={clip_prompt_similarity:.2f}; "
                f"kind={kind}"
            )

            scored.append(
                (
                    ClipPlan(
                        start=window_start,
                        end=window_end,
                        title_hint=hint,
                        visual_score=score_10,
                        attempt_id=attempt.attempt_id,
                        kind=kind,
                    ),
                    score_10,
                )
            )

        if not scored:
            continue

        scored.sort(key=lambda item: item[1], reverse=True)
        primary_candidates.append(scored[0])
        for extra in scored[1:]:
            secondary_candidates.append(extra)

    primary_candidates.sort(key=lambda item: item[1], reverse=True)
    secondary_candidates.sort(key=lambda item: item[1], reverse=True)

    chosen: list[ClipPlan] = [item[0] for item in primary_candidates]

    if len(chosen) < SETTINGS.shorts_per_video:
        for candidate, score in secondary_candidates:
            if score < 5.5:
                continue
            if any(candidate.attempt_id == existing.attempt_id and candidate.kind == existing.kind for existing in chosen):
                continue
            if any(overlaps(candidate, existing) for existing in chosen):
                continue
            chosen.append(candidate)
            if len(chosen) >= SETTINGS.shorts_per_video:
                break

    chosen.sort(key=lambda plan: plan.visual_score, reverse=True)
    return dedupe_clip_plans(chosen)[:SETTINGS.shorts_per_video]


def attempt_highlight_plans(
    video_path: Path,
    segments: list[Segment],
    progress_cb: Callable[[int, str], None] | None = None,
) -> list[ClipPlan]:
    def step(progress: int, message: str) -> None:
        if progress_cb is not None:
            progress_cb(progress, message)

    step(28, "Filtering scenes")
    scene_spans = detect_scene_spans(video_path)

    step(32, f"Tracking climber + pose across {len(scene_spans)} scene(s)")
    samples, meta = run_tracking_pose(
        video_path,
        scene_spans,
        progress_cb=lambda current, total: step(
            32 + int((current / max(1, total)) * 28),
            f"Tracking climber + pose (frame {current:,}/{total:,})",
        ),
    )

    if not samples:
        step(70, "No tracked climber found")
        return []

    step(62, "Detecting attempts")
    grouped: dict[int, list[FrameSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.scene_idx, []).append(sample)

    all_attempt_plans: list[ClipPlan] = []

    for scene_idx in sorted(grouped.keys()):
        scene_samples = grouped[scene_idx]
        attempts, metrics = detect_attempts(scene_samples, float(meta["fps"]))
        if not attempts:
            continue
        plans = score_attempt_candidates(video_path, scene_samples, attempts, metrics, segments)
        all_attempt_plans.extend(plans)

    all_attempt_plans.sort(key=lambda plan: plan.visual_score, reverse=True)
    step(70, f"Selected {len(all_attempt_plans)} candidate highlight(s)")
    return all_attempt_plans[:SETTINGS.shorts_per_video]


def build_caption_chunks(
    segments: list[Segment],
    clip_start: float,
    clip_end: float,
    max_words: int | None = None,
    max_duration: float | None = None,
    max_gap: float | None = None,
) -> list[tuple[float, float, str]]:
    max_words = max_words or SETTINGS.caption_max_words
    max_duration = max_duration or SETTINGS.caption_max_duration
    max_gap = max_gap or SETTINGS.caption_max_gap

    words: list[Word] = []

    for segment in segments:
        if segment.end <= clip_start or segment.start >= clip_end:
            continue

        if segment.words:
            for word in segment.words:
                if word.end <= clip_start or word.start >= clip_end:
                    continue
                token = word.text.strip()
                if not token:
                    continue
                words.append(
                    Word(
                        start=max(word.start, clip_start) - clip_start,
                        end=min(word.end, clip_end) - clip_start,
                        text=token,
                    )
                )
        else:
            raw_words = [w for w in segment.text.split() if w.strip()]
            if not raw_words:
                continue

            seg_start = max(segment.start, clip_start)
            seg_end = min(segment.end, clip_end)
            seg_duration = max(0.01, seg_end - seg_start)
            per_word = seg_duration / max(1, len(raw_words))

            for i, token in enumerate(raw_words):
                start = seg_start + (i * per_word)
                end = start + per_word
                words.append(
                    Word(
                        start=start - clip_start,
                        end=end - clip_start,
                        text=token.strip(),
                    )
                )

    words = [word for word in words if word.text]
    if not words:
        return []

    chunks: list[tuple[float, float, str]] = []
    current_words: list[str] = []
    current_start: float | None = None
    current_end: float | None = None

    def flush() -> None:
        nonlocal current_words, current_start, current_end
        if not current_words or current_start is None or current_end is None:
            return
        chunks.append((current_start, current_end, " ".join(current_words)))
        current_words = []
        current_start = None
        current_end = None

    for word in words:
        should_flush = False

        if current_words:
            if len(current_words) >= max_words:
                should_flush = True
            elif current_end is not None and (word.start - current_end) > max_gap:
                should_flush = True
            elif current_start is not None and (word.end - current_start) > max_duration:
                should_flush = True
            elif current_words[-1].endswith((".", "!", "?", ",")):
                should_flush = True

        if should_flush:
            flush()

        if current_start is None:
            current_start = word.start

        current_words.append(word.text)
        current_end = word.end

    flush()
    return chunks


def ass_time(value: float) -> str:
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = value % 60
    return f"{hours}:{minutes:02d}:{seconds:05.2f}".replace(".", ",")


def write_ass(subtitles: list[tuple[float, float, str]], path: Path) -> None:
    style = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {SETTINGS.render_width}
PlayResY: {SETTINGS.render_height}

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,62,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,2,10,10,220,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    lines = [style]
    for start, end, text in subtitles:
        clean = text.replace("{", "(").replace("}", ")").replace("\n", " ").strip()
        if not clean:
            continue
        lines.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{clean}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_short(source_path: Path, out_path: Path, thumb_path: Path, start_sec: float, end_sec: float, ass_path: Path | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)

    vf_parts = [
        f"scale={SETTINGS.render_width}:{SETTINGS.render_height}:force_original_aspect_ratio=increase",
        f"crop={SETTINGS.render_width}:{SETTINGS.render_height}",
    ]

    if ass_path is not None:
        ass_value = ass_path.resolve().as_posix().replace(":", "\\:").replace("'", "\\'")
        vf_parts.append(f"subtitles='{ass_value}'")

    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_sec),
            "-to",
            str(end_sec),
            "-i",
            str(source_path),
            "-vf",
            ",".join(vf_parts),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "ffmpeg render failed")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_sec + 0.6),
            "-i",
            str(source_path),
            "-vframes",
            "1",
            "-vf",
            "scale=640:-1",
            str(thumb_path),
        ],
        capture_output=True,
        text=True,
    )


def process_video(
    video_id: str,
    raw_path: Path,
    progress_cb: Callable[[int, str], None] | None = None,
) -> list[dict[str, Any]]:
    def step(progress: int, message: str) -> None:
        if progress_cb is not None:
            progress_cb(progress, message)

    step(10, "Checking for speech")
    speech_ratio = speech_ratio_from_video(raw_path)

    step(16, "Deciding caption/transcript mode")
    use_transcript = should_use_transcript(speech_ratio)
    use_captions = captions_enabled(speech_ratio)

    segments: list[Segment] = []
    if use_transcript:
        step(24, "Transcribing speech")
        segments = transcribe_video(raw_path)

    transcript_path = SETTINGS.transcripts_dir / f"{video_id}.json"
    transcript_path.write_text(
        json.dumps([segment_to_dict(seg) for seg in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    visual_plans: list[ClipPlan] = []
    transcript_plans: list[ClipPlan] = []

    if SETTINGS.moment_mode in {"visual", "hybrid"}:
        step(28, "Finding attempt-aware highlights")
        visual_plans = attempt_highlight_plans(video_path=raw_path, segments=segments, progress_cb=step)

    if SETTINGS.moment_mode in {"transcript", "hybrid"} and segments:
        step(70, "Building transcript fallback")
        transcript_plans = transcript_clip_plans(segments)

    if SETTINGS.moment_mode == "transcript":
        plans = transcript_plans
    elif SETTINGS.moment_mode == "hybrid":
        plans = visual_plans if visual_plans else transcript_plans
    else:
        plans = visual_plans

    if not plans:
        step(100, "No highlights found")
        return []

    step(72, "Rendering highlights")
    shorts: list[dict[str, Any]] = []
    total = min(len(plans), SETTINGS.shorts_per_video)

    for index, plan in enumerate(plans[:SETTINGS.shorts_per_video], start=1):
        render_progress = 72 + int(((index - 1) / max(1, total)) * 20)
        step(render_progress, f"Rendering short {index}/{total}")

        clip_id = uuid.uuid4().hex
        out_path = SETTINGS.shorts_dir / f"{video_id}_{index:02d}_{clip_id}.mp4"
        thumb_path = SETTINGS.thumbs_dir / f"{video_id}_{index:02d}_{clip_id}.jpg"

        clip_segments = segments_in_window(segments, plan.start, plan.end)
        transcript_text = extract_window_text(segments, plan.start, plan.end)

        ass_path: Path | None = None
        if use_captions and clip_segments:
            caption_chunks = build_caption_chunks(
                clip_segments,
                plan.start,
                plan.end,
            )
            if caption_chunks:
                ass_path = SETTINGS.shorts_dir / f"{video_id}_{index:02d}_{clip_id}.ass"
                write_ass(caption_chunks, ass_path)

        render_short(raw_path, out_path, thumb_path, plan.start, plan.end, ass_path)

        shorts.append(
            {
                "id": clip_id,
                "start_sec": round(plan.start, 2),
                "end_sec": round(plan.end, 2),
                "score": round(plan.visual_score, 2),
                "reason": (
                    f"speech_ratio={speech_ratio:.3f}; "
                    f"captions={use_captions}; "
                    f"{plan.title_hint}"
                ),
                "short_path": safe_rel(out_path),
                "thumb_path": safe_rel(thumb_path),
                "transcript": transcript_text,
                "created_at": utc_now(),
            }
        )

    step(94, "Ranking highlights")
    shorts.sort(key=lambda item: item["score"], reverse=True)
    for idx, item in enumerate(shorts, start=1):
        item["idx"] = idx

    step(100, "Short generation complete")
    return shorts


def watcher_loop() -> None:
    init_db()
    SETTINGS.ensure_dirs()
    log_event("INFO", "watcher", "Watcher started.")

    while True:
        try:
            items = list_playlist_items()
            added = 0

            with db() as conn:
                existing = {row["id"] for row in conn.execute("SELECT id FROM videos").fetchall()}
                for item in items:
                    if item["video_id"] in existing:
                        continue

                    conn.execute(
                        """
                        INSERT INTO videos (id, playlist_id, title, channel_title, added_at, status, raw_path, error)
                        VALUES (?, ?, ?, ?, ?, 'new', '', '')
                        """,
                        (
                            item["video_id"],
                            SETTINGS.youtube_playlist_id,
                            item["title"],
                            item["channel_title"],
                            item["added_at"],
                        ),
                    )
                    conn.commit()

                    enqueue_job("download", {"video_id": item["video_id"]})
                    enqueue_job("shortify", {"video_id": item["video_id"]})
                    added += 1

            if added > 0:
                log_event("INFO", "watcher", f"Queued {added} new video(s).")
            else:
                log_event("INFO", "watcher", "No new videos found.")

            time.sleep(max(15, SETTINGS.poll_interval_seconds))
        except Exception as exc:
            log_event("ERROR", "watcher", str(exc))
            print(f"[watcher] {exc}")
            time.sleep(max(15, SETTINGS.poll_interval_seconds))


def worker_loop() -> None:
    init_db()
    SETTINGS.ensure_dirs()
    recover_interrupted_jobs()
    log_event("INFO", "worker", "Worker started.")

    while True:
        job = next_queued_job()
        if job is None:
            time.sleep(1.0)
            continue

        payload = json.loads(job["payload_json"])
        update_job(job["id"], status="running", progress=1, message="Starting")

        try:
            if job["type"] == "download":
                video_id = payload["video_id"]
                log_event("INFO", "download", f"Starting download for {video_id}", job_id=job["id"], video_id=video_id)
                update_job(job["id"], progress=10, message="Downloading")
                path = download_video(video_id)

                with db() as conn:
                    conn.execute(
                        "UPDATE videos SET status = 'downloaded', raw_path = ?, error = '' WHERE id = ?",
                        (safe_rel(path), video_id),
                    )
                    conn.commit()

                update_job(job["id"], status="done", progress=100, message="Downloaded")
                log_event("INFO", "download", f"Download complete for {video_id}", job_id=job["id"], video_id=video_id)

            elif job["type"] == "shortify":
                video_id = payload["video_id"]
                with db() as conn:
                    row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()

                if row is None:
                    raise RuntimeError("Video not found.")

                if not row["raw_path"]:
                    update_job(job["id"], status="queued", progress=0, message="Waiting for download")
                    time.sleep(0.5)
                    continue

                raw_path = resolve_media_path(row["raw_path"])
                log_event("INFO", "shortify", f"Starting short generation for {video_id}", job_id=job["id"], video_id=video_id)

                def on_progress(progress: int, message: str) -> None:
                    update_job(job["id"], progress=progress, message=message)

                shorts = process_video(video_id, raw_path, progress_cb=on_progress)

                with db() as conn:
                    conn.execute("DELETE FROM shorts WHERE video_id = ?", (video_id,))
                    for short in shorts:
                        conn.execute(
                            """
                            INSERT INTO shorts (id, video_id, idx, start_sec, end_sec, score, reason, status, short_path, thumb_path, transcript, uploaded_id, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?)
                            """,
                            (
                                short["id"],
                                video_id,
                                short["idx"],
                                short["start_sec"],
                                short["end_sec"],
                                short["score"],
                                short["reason"],
                                "generated",
                                short["short_path"],
                                short["thumb_path"],
                                short["transcript"],
                                short["created_at"],
                            ),
                        )

                    conn.execute("UPDATE videos SET status = 'processed', error = '' WHERE id = ?", (video_id,))
                    conn.commit()

                if SETTINGS.auto_select_top_n > 0:
                    with db() as conn:
                        chosen = conn.execute(
                            """
                            SELECT id FROM shorts
                            WHERE video_id = ?
                            ORDER BY score DESC
                            LIMIT ?
                            """,
                            (video_id, SETTINGS.auto_select_top_n),
                        ).fetchall()
                        for row_short in chosen:
                            conn.execute(
                                "UPDATE shorts SET status = 'selected' WHERE id = ?",
                                (row_short["id"],),
                            )
                        conn.commit()

                update_job(job["id"], status="done", progress=100, message="Shorts generated")
                log_event(
                    "INFO",
                    "shortify",
                    f"Short generation complete for {video_id} ({len(shorts)} shorts)",
                    job_id=job["id"],
                    video_id=video_id,
                )

            else:
                raise RuntimeError(f"Unknown job type: {job['type']}")

        except Exception as exc:
            error_text = str(exc)[:400]

            with db() as conn:
                if job["type"] == "download":
                    video_id = payload.get("video_id", "")
                    if video_id:
                        conn.execute(
                            "UPDATE videos SET status = 'error', error = ? WHERE id = ?",
                            (error_text, video_id),
                        )
                elif job["type"] == "shortify":
                    video_id = payload.get("video_id", "")
                    if video_id:
                        conn.execute(
                            "UPDATE videos SET status = 'error', error = ? WHERE id = ?",
                            (error_text, video_id),
                        )
                conn.commit()

            update_job(job["id"], status="error", progress=100, message=error_text)
            log_event(
                "ERROR",
                job["type"],
                error_text,
                job_id=job["id"],
                video_id=payload.get("video_id", ""),
            )


def shell_html(title: str, body: str) -> str:
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{
    margin: 0;
    font-family: Arial, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
}}
a {{ color: #93c5fd; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
nav {{
    background: #111827;
    border-bottom: 1px solid #1f2937;
    padding: 14px 22px;
}}
nav a {{
    margin-right: 18px;
}}
.container {{
    max-width: 1280px;
    margin: 0 auto;
    padding: 24px;
}}
.card {{
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 18px;
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
}}
th, td {{
    text-align: left;
    padding: 12px;
    border-top: 1px solid #1f2937;
    vertical-align: top;
}}
.badge {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    background: #1f2937;
    font-size: 12px;
}}
.gallery {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 16px;
}}
.thumb {{
    width: 100%;
    aspect-ratio: 9 / 16;
    object-fit: cover;
    border-radius: 12px;
    background: #000;
}}
button {{
    background: #2563eb;
    color: #fff;
    border: none;
    padding: 10px 14px;
    border-radius: 10px;
    cursor: pointer;
}}
button.secondary {{
    background: #374151;
}}
small, .muted {{ color: #94a3b8; }}
pre {{
    white-space: pre-wrap;
    word-break: break-word;
}}
input, textarea {{
    width: 100%;
    background: #0b1220;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 10px;
}}
</style>
</head>
<body>
<nav>
  <a href="/">Dashboard</a>
  <a href="/videos">Videos</a>
  <a href="/shorts">Shorts</a>
  <a href="/queue">Queue</a>
</nav>
<div class="container">
{body}
</div>
</body>
</html>"""


app = FastAPI(title="Shorts Factory")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    SETTINGS.ensure_dirs()


@app.get("/api/dashboard")
def dashboard_api() -> dict[str, Any]:
    with db() as conn:
        videos_count = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        shorts_count = conn.execute("SELECT COUNT(*) FROM shorts").fetchone()[0]
        active = conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')").fetchone()[0]
        jobs = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 20").fetchall()
        logs = conn.execute("SELECT * FROM app_logs ORDER BY id DESC LIMIT 30").fetchall()

    return {
        "videos_count": videos_count,
        "shorts_count": shorts_count,
        "active": active,
        "jobs": [
            {
                "created_at": row["created_at"],
                "type": row["type"],
                "status": row["status"],
                "progress": row["progress"],
                "message": row["message"],
            }
            for row in jobs
        ],
        "logs": [
            {
                "created_at": row["created_at"],
                "level": row["level"],
                "source": row["source"],
                "message": row["message"],
            }
            for row in logs
        ],
    }


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    body = """
    <div class="grid">
      <div class="card">
        <div class="muted">Videos</div>
        <h2 id="videos-count">0</h2>
      </div>
      <div class="card">
        <div class="muted">Shorts</div>
        <h2 id="shorts-count">0</h2>
      </div>
      <div class="card">
        <div class="muted">Active jobs</div>
        <h2 id="active-count">0</h2>
      </div>
    </div>

    <div class="card" style="margin-top:18px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h2>Recent jobs</h2>
        <small id="dashboard-status" class="muted">Loading...</small>
      </div>
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Type</th>
            <th>Status</th>
            <th>Progress</th>
            <th>Message</th>
          </tr>
        </thead>
        <tbody id="jobs-body"></tbody>
      </table>
    </div>

    <div class="card" style="margin-top:18px;">
      <h2>Recent logs</h2>
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Level</th>
            <th>Source</th>
            <th>Message</th>
          </tr>
        </thead>
        <tbody id="logs-body"></tbody>
      </table>
    </div>

    <script>
    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function renderJobs(rows) {
      const body = document.getElementById("jobs-body");
      if (!rows || rows.length === 0) {
        body.innerHTML = "<tr><td colspan='5' class='muted'>No jobs yet.</td></tr>";
        return;
      }

      body.innerHTML = rows.map((job) => `
        <tr>
          <td>${escapeHtml(job.created_at)}</td>
          <td>${escapeHtml(job.type)}</td>
          <td>${escapeHtml(job.status)}</td>
          <td>${escapeHtml(job.progress)}%</td>
          <td>${escapeHtml(job.message)}</td>
        </tr>
      `).join("");
    }

    function renderLogs(rows) {
      const body = document.getElementById("logs-body");
      if (!rows || rows.length === 0) {
        body.innerHTML = "<tr><td colspan='4' class='muted'>No logs yet.</td></tr>";
        return;
      }

      body.innerHTML = rows.map((log) => `
        <tr>
          <td>${escapeHtml(log.created_at)}</td>
          <td>${escapeHtml(log.level)}</td>
          <td>${escapeHtml(log.source)}</td>
          <td>${escapeHtml(log.message)}</td>
        </tr>
      `).join("");
    }

    async function refreshDashboard() {
      const statusEl = document.getElementById("dashboard-status");

      try {
        const response = await fetch("/api/dashboard", { cache: "no-store" });
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }

        const data = await response.json();

        document.getElementById("videos-count").textContent = data.videos_count;
        document.getElementById("shorts-count").textContent = data.shorts_count;
        document.getElementById("active-count").textContent = data.active;

        renderJobs(data.jobs);
        renderLogs(data.logs);

        statusEl.textContent = "Live";
      } catch (error) {
        statusEl.textContent = "Update failed";
        console.error(error);
      }
    }

    refreshDashboard();
    setInterval(refreshDashboard, 2500);
    </script>
    """
    return shell_html("Dashboard", body)


@app.get("/queue", response_class=HTMLResponse)
def queue_page() -> str:
    with db() as conn:
        jobs = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 200").fetchall()

    rows = []
    for job in jobs:
        rows.append(
            f"<tr><td>{html.escape(job['created_at'])}</td><td>{html.escape(job['id'][:8])}</td><td>{html.escape(job['type'])}</td><td>{html.escape(job['status'])}</td><td>{job['progress']}%</td><td>{html.escape(job['message'])}</td></tr>"
        )

    body = f"""
    <div class="card">
      <h2>Queue</h2>
      <table>
        <thead><tr><th>Time</th><th>Job</th><th>Type</th><th>Status</th><th>Progress</th><th>Message</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """
    return shell_html("Queue", body)


@app.get("/videos", response_class=HTMLResponse)
def videos_page() -> str:
    with db() as conn:
        videos = conn.execute("SELECT * FROM videos ORDER BY added_at DESC LIMIT 200").fetchall()

    rows = []
    for video in videos:
        rows.append(
            f"<tr>"
            f"<td>{html.escape(video['added_at'])}</td>"
            f"<td><a href='/video/{html.escape(video['id'])}'>{html.escape(video['title'])}</a><br><small>{html.escape(video['id'])}</small></td>"
            f"<td>{html.escape(video['status'])}</td>"
            f"<td>{html.escape(video['channel_title'])}</td>"
            f"</tr>"
        )

    body = f"""
    <div class="card">
      <h2>Videos</h2>
      <table>
        <thead><tr><th>Added</th><th>Video</th><th>Status</th><th>Channel</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """
    return shell_html("Videos", body)


@app.get("/video/{video_id}", response_class=HTMLResponse)
def video_page(video_id: str) -> str:
    with db() as conn:
        video = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        shorts = conn.execute("SELECT * FROM shorts WHERE video_id = ? ORDER BY score DESC", (video_id,)).fetchall()

    if video is None:
        return shell_html("Not Found", "<div class='card'><h2>Video not found.</h2></div>")

    cards = []
    for short in shorts:
        thumb_url = f"/media?{urlencode({'path': short['thumb_path']})}"
        video_url = f"/media?{urlencode({'path': short['short_path']})}"
        toggle_url = f"/short/{short['id']}/toggle"
        button_label = "Selected" if short["status"] == "selected" else "Select"
        button_class = "secondary" if short["status"] == "selected" else ""
        cards.append(
            f"""
            <div class="card">
              <img class="thumb" src="{thumb_url}" alt="">
              <h3>#{short['idx']} — score {short['score']:.2f}</h3>
              <div class="muted">{html.escape(short['reason'])}</div>
              <div style="margin-top:12px;"><a href="{video_url}">Preview</a></div>
              <form method="post" action="{toggle_url}" style="margin-top:12px;">
                <button class="{button_class}" type="submit">{button_label}</button>
              </form>
            </div>
            """
        )

    body = f"""
    <div class="card">
      <h2>{html.escape(video['title'])}</h2>
      <div class="muted">{html.escape(video['id'])} • {html.escape(video['channel_title'])}</div>
      <div style="margin-top:10px;"><span class="badge">{html.escape(video['status'])}</span></div>
      {f"<pre style='margin-top:12px'>{html.escape(video['error'])}</pre>" if video['error'] else ""}
    </div>

    <div class="gallery" style="margin-top:18px;">
      {''.join(cards) if cards else "<div class='card'>No shorts yet.</div>"}
    </div>
    """
    return shell_html(video["title"], body)


@app.get("/shorts", response_class=HTMLResponse)
def shorts_page() -> str:
    with db() as conn:
        shorts = conn.execute("SELECT * FROM shorts ORDER BY score DESC, created_at DESC LIMIT 300").fetchall()

    cards = []
    for short in shorts:
        thumb_url = f"/media?{urlencode({'path': short['thumb_path']})}"
        video_url = f"/media?{urlencode({'path': short['short_path']})}"
        toggle_url = f"/short/{short['id']}/toggle"
        button_label = "Selected" if short["status"] == "selected" else "Select"
        button_class = "secondary" if short["status"] == "selected" else ""
        cards.append(
            f"""
            <div class="card">
              <img class="thumb" src="{thumb_url}" alt="">
              <h3><a href="/video/{html.escape(short['video_id'])}">Video</a> — #{short['idx']}</h3>
              <div>Score {short['score']:.2f}</div>
              <div class="muted" style="margin-top:6px;">{html.escape(short['reason'])}</div>
              <div style="margin-top:12px;"><a href="{video_url}">Preview</a></div>
              <form method="post" action="{toggle_url}" style="margin-top:12px;">
                <button class="{button_class}" type="submit">{button_label}</button>
              </form>
            </div>
            """
        )

    body = f"<div class='gallery'>{''.join(cards)}</div>"
    return shell_html("Shorts", body)


@app.post("/short/{short_id}/toggle")
def toggle_short(short_id: str) -> RedirectResponse:
    with db() as conn:
        row = conn.execute("SELECT * FROM shorts WHERE id = ?", (short_id,)).fetchone()
        if row is not None:
            new_status = "generated" if row["status"] == "selected" else "selected"
            conn.execute("UPDATE shorts SET status = ? WHERE id = ?", (new_status, short_id))
            conn.commit()
            return RedirectResponse(f"/video/{row['video_id']}", status_code=303)
    return RedirectResponse("/shorts", status_code=303)


@app.get("/media")
def media(path: str):
    resolved = resolve_media_path(path)
    if not resolved.exists() or not resolved.is_file():
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(str(resolved))


def print_usage() -> None:
    print("Usage:")
    print("  python main.py init")
    print("  python main.py web")
    print("  python main.py watch")
    print("  python main.py work")
    print("  python main.py all")


def run_all() -> None:
    init_db()
    SETTINGS.ensure_dirs()

    worker_thread = threading.Thread(target=worker_loop, name="worker", daemon=True)
    watcher_thread = threading.Thread(target=watcher_loop, name="watcher", daemon=True)

    worker_thread.start()
    watcher_thread.start()

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)


def main() -> None:
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()

    if command == "init":
        init_db()
        SETTINGS.ensure_dirs()
        print("Initialised.")
        return

    if command == "watch":
        watcher_loop()
        return

    if command == "work":
        worker_loop()
        return

    if command == "web":
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
        return

    if command == "all":
        run_all()
        return

    print_usage()


if __name__ == "__main__":
    main()