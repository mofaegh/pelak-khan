#!/usr/bin/env python
"""Pelak-Khan FastAPI backend v0.4.

Features:
- image ALPR
- temporal live-camera ALPR for moving vehicles
- asynchronous video upload / processing
- OCR temporal voting + sharpest-frame evidence
- SQLite history/search/delete/backup APIs
"""

from __future__ import annotations

import math
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pelak_khan.postprocessing.plate_regions import lookup_plate_region
from pelak_khan.postprocessing.plate_validator import display_plate, expand_box
from pelak_khan.runtime import RecognitionService
from pelak_khan.storage.database import (
    clear_database,
    connect,
    create_backup,
    delete_detection,
    delete_plate,
    ingest_records,
    is_media_referenced,
)
from pelak_khan.tracking import TemporalPlateTracker, TemporalTrack


DEFAULT_APP_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(os.getenv("PELAK_APP_ROOT", str(DEFAULT_APP_ROOT)))
DB_PATH = Path(os.getenv("PELAK_DB_PATH", str(APP_ROOT / "data" / "pelak_khan.db")))
FRONTEND_DIR = Path(os.getenv("PELAK_FRONTEND_DIR", str(APP_ROOT / "apps" / "frontend")))
STORAGE_ROOT = Path(os.getenv("PELAK_STORAGE_ROOT", str(APP_ROOT / "data" / "storage")))
DETECTOR_PATH = Path(
    os.getenv("PELAK_DETECTOR_PATH", str(APP_ROOT / "models" / "runtime" / "detector_v1.pt"))
)
OCR_PATH = Path(
    os.getenv("PELAK_OCR_PATH", str(APP_ROOT / "models" / "runtime" / "ocr_v1.pt"))
)
DEVICE = os.getenv("PELAK_DEVICE", "auto")
MAX_UPLOAD_MB = int(os.getenv("PELAK_MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_VIDEO_MB = int(os.getenv("PELAK_MAX_VIDEO_MB", "750"))
MAX_VIDEO_BYTES = MAX_VIDEO_MB * 1024 * 1024
LIVE_DEDUP_SECONDS = float(os.getenv("PELAK_LIVE_DEDUP_SECONDS", "12"))
VIDEO_DEDUP_SECONDS = float(os.getenv("PELAK_VIDEO_DEDUP_SECONDS", "10"))
LIVE_SESSION_TTL = float(os.getenv("PELAK_LIVE_SESSION_TTL", "900"))

ORIGINALS_DIR = STORAGE_ROOT / "originals"
CROPS_DIR = STORAGE_ROOT / "crops"
ANNOTATED_DIR = STORAGE_ROOT / "annotated"
VIDEOS_DIR = STORAGE_ROOT / "videos"
BACKUPS_DIR = Path(os.getenv("PELAK_BACKUPS_DIR", str(APP_ROOT / "data" / "backups")))
for directory in (ORIGINALS_DIR, CROPS_DIR, ANNOTATED_DIR, VIDEOS_DIR, BACKUPS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

with connect(DB_PATH):
    pass

app = FastAPI(
    title="Pelak-Khan API",
    version="0.5.0",
    description="Iranian license plate recognition backend",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(STORAGE_ROOT)), name="media")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

_service: RecognitionService | None = None
_service_lock = threading.Lock()
_inference_lock = threading.Lock()
_live_lock = threading.Lock()
_live_sessions: dict[str, dict[str, Any]] = {}
_video_lock = threading.Lock()
_video_jobs: dict[str, dict[str, Any]] = {}


def get_service() -> RecognitionService:
    global _service
    if _service is not None:
        return _service
    with _service_lock:
        if _service is None:
            try:
                _service = RecognitionService(
                    detector_path=DETECTOR_PATH,
                    ocr_path=OCR_PATH,
                    device=DEVICE,
                )
            except (FileNotFoundError, KeyError, RuntimeError) as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _service


def rows_to_dicts(rows) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def media_url(relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    clean = relative_path.replace("\\", "/").lstrip("/")
    return f"/media/{clean}"


def with_region(item: dict[str, Any], text_key: str = "raw_text") -> dict[str, Any]:
    output = dict(item)
    text = str(output.get(text_key) or output.get("plate_text") or "")
    output["region"] = lookup_plate_region(text)
    return output


def new_event_id(prefix: str = "") -> str:
    now = datetime.now(timezone.utc)
    core = f"{now.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:12]}"
    return f"{prefix}{core}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def read_image_upload(file: UploadFile) -> tuple[bytes, np.ndarray]:
    content_type = (file.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported.")

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image is larger than {MAX_UPLOAD_MB} MB.")

    encoded = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image file.")
    return payload, frame


async def save_video_upload(file: UploadFile, destination: Path) -> int:
    content_type = (file.content_type or "").lower()
    suffix = Path(file.filename or "").suffix.lower()
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
    if suffix not in allowed and content_type and not content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail="Unsupported video upload.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_VIDEO_BYTES:
                    raise HTTPException(status_code=413, detail=f"Video is larger than {MAX_VIDEO_MB} MB.")
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")
    return total


def _safe_storage_path(relative_path: str) -> Path | None:
    if not relative_path:
        return None
    root = STORAGE_ROOT.resolve()
    candidate = (STORAGE_ROOT / relative_path.replace("\\", "/").lstrip("/")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _remove_media_if_unused(relative_path: str) -> bool:
    if not relative_path or is_media_referenced(DB_PATH, relative_path):
        return False
    path = _safe_storage_path(relative_path)
    if path is None or not path.is_file():
        return False
    path.unlink(missing_ok=True)
    return True


def _cleanup_detection_media(source_image: str, crop_path: str) -> int:
    removed = 0
    if crop_path and _remove_media_if_unused(crop_path):
        removed += 1
    source_removed = source_image and _remove_media_if_unused(source_image)
    if source_removed:
        removed += 1
        source_name = Path(source_image).name
        annotated_relative = f"annotated/{Path(source_name).stem}.jpg"
        annotated = _safe_storage_path(annotated_relative)
        if annotated and annotated.exists():
            annotated.unlink(missing_ok=True)
            removed += 1
    return removed


def _draw_evidence(frame: np.ndarray, record: dict[str, Any], path: Path) -> None:
    annotated = frame.copy()
    x1, y1, x2, y2 = (float(record.get(k, 0.0)) for k in ("x1", "y1", "x2", "y2"))
    cv2.rectangle(
        annotated,
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        (0, 200, 0),
        2,
    )
    label = f"TRACK {record.get('track_id', '')} det={float(record.get('det_confidence', 0.0)):.3f}"
    cv2.putText(
        annotated,
        label,
        (int(round(x1)), max(20, int(round(y1)) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 200, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), annotated)


def persist_temporal_track(
    track: TemporalTrack,
    *,
    prefix: str,
    source_type: str,
    source_ref: str,
) -> dict[str, Any] | None:
    frame = track.best_frame
    record = track.snapshot(reason="persisted")
    text = str(record.get("raw_text", "")).strip()
    if frame is None or not text or int(record.get("temporal_valid_hits", 0) or 0) < 1:
        return None

    event_id = new_event_id(prefix)
    original_name = f"{event_id}.jpg"
    original_relative = f"originals/{original_name}"
    original_path = ORIGINALS_DIR / original_name
    cv2.imwrite(str(original_path), frame)

    annotated_name = f"{event_id}.jpg"
    annotated_relative = f"annotated/{annotated_name}"
    _draw_evidence(frame, record, ANNOTATED_DIR / annotated_name)

    height, width = frame.shape[:2]
    service = get_service()
    x1, y1, x2, y2 = (float(record.get(k, 0.0)) for k in ("x1", "y1", "x2", "y2"))
    cx1, cy1, cx2, cy2 = expand_box(x1, y1, x2, y2, width, height, service.pad_frac)
    crop = frame[cy1:cy2, cx1:cx2]
    crop_name = f"{event_id}_plate_01.jpg"
    crop_relative = f"crops/{crop_name}"
    if crop.size:
        cv2.imwrite(str(CROPS_DIR / crop_name), crop)
    else:
        crop_relative = ""

    record.update(
        {
            "source_image": original_relative,
            "crop_path": crop_relative,
            "display_text": display_plate(text),
            "accepted": True,
            "status": "ACCEPTED_TEMPORAL",
            "source_type": source_type,
            "source_ref": source_ref,
        }
    )
    db_counts = ingest_records(DB_PATH, [record])
    output = with_region(record)
    output["crop_url"] = media_url(crop_relative)
    output["source_url"] = media_url(original_relative)
    output["annotated_url"] = media_url(annotated_relative)
    return {
        "event_id": event_id,
        "database": db_counts,
        "result": output,
        "original_url": media_url(original_relative),
        "annotated_url": media_url(annotated_relative),
    }


def _cleanup_live_sessions() -> None:
    now = time.monotonic()
    stale = [sid for sid, state in _live_sessions.items() if now - float(state["last_access"]) > LIVE_SESSION_TTL]
    for sid in stale:
        _live_sessions.pop(sid, None)


def _get_live_session(session_id: str, min_hits: int) -> dict[str, Any]:
    with _live_lock:
        _cleanup_live_sessions()
        state = _live_sessions.get(session_id)
        if state is None:
            state = {
                "tracker": TemporalPlateTracker(min_hits=min_hits, max_missed_frames=2),
                "frame_index": 0,
                "last_access": time.monotonic(),
                "last_saved": {},
            }
            _live_sessions[session_id] = state
        state["last_access"] = time.monotonic()
        state["tracker"].min_hits = max(1, min_hits)
        return state


def _live_can_save(state: dict[str, Any], text: str) -> bool:
    now = time.monotonic()
    previous = state["last_saved"].get(text)
    if previous is not None and now - previous < LIVE_DEDUP_SECONDS:
        return False
    state["last_saved"][text] = now
    return True


def _sum_db_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(FRONTEND_DIR / "icon.svg", media_type="image/svg+xml")


@app.get("/api/info")
def api_info():
    return {
        "name": "Pelak-Khan API",
        "version": app.version,
        "docs": "/docs",
        "recognize": "/api/recognize/image",
        "live_recognize": "/api/recognize/live-frame",
        "video_recognize": "/api/recognize/video",
        "ui": "/",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": app.version,
        "database": str(DB_PATH),
        "database_exists": DB_PATH.exists(),
        "detector": str(DETECTOR_PATH),
        "detector_exists": DETECTOR_PATH.exists(),
        "ocr": str(OCR_PATH),
        "ocr_exists": OCR_PATH.exists(),
        "storage": str(STORAGE_ROOT),
        "device": DEVICE,
        "live_dedup_seconds": LIVE_DEDUP_SECONDS,
        "video_dedup_seconds": VIDEO_DEDUP_SECONDS,
        "max_video_mb": MAX_VIDEO_MB,
        "video_jobs": len(_video_jobs),
    }


@app.post("/api/recognize/image")
async def recognize_image(
    file: UploadFile = File(...),
    det_conf: float = Form(0.25, ge=0.01, le=0.99),
    allow_edge: bool = Form(False),
):
    payload, frame = await read_image_upload(file)
    event_id = new_event_id()

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        suffix = ".jpg"

    original_name = f"{event_id}{suffix}"
    original_path = ORIGINALS_DIR / original_name
    original_path.write_bytes(payload)
    original_relative = f"originals/{original_name}"

    annotated_name = f"{event_id}.jpg"
    annotated_path = ANNOTATED_DIR / annotated_name
    annotated_relative = f"annotated/{annotated_name}"

    try:
        with _inference_lock:
            result = get_service().recognize_frame(
                frame,
                source_image=original_relative,
                crops_dir=CROPS_DIR,
                crop_path_prefix=event_id,
                annotated_path=annotated_path,
                annotated_relative_path=annotated_relative,
                det_conf=det_conf,
                allow_edge=allow_edge,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {type(exc).__name__}: {exc}") from exc

    for item in result["results"]:
        item["source_type"] = "image"
        item["source_ref"] = original_relative
    db_counts = ingest_records(DB_PATH, result["results"])

    response_results = []
    for item in result["results"]:
        output = with_region(item)
        output["crop_url"] = media_url(str(item.get("crop_path", "")))
        response_results.append(output)

    return {
        "success": True,
        "mode": "image",
        "event_id": event_id,
        "original_url": media_url(original_relative),
        "annotated_url": media_url(annotated_relative),
        "detections": result["detections"],
        "accepted": result["accepted"],
        "rejected": result["rejected"],
        "database": db_counts,
        "results": response_results,
    }


@app.post("/api/recognize/live-frame")
async def recognize_live_frame(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
    det_conf: float = Form(0.25, ge=0.01, le=0.99),
    allow_edge: bool = Form(False),
    min_hits: int = Form(2, ge=1, le=5),
):
    """Process a sampled browser-camera frame with tracking and temporal OCR voting."""
    _, frame = await read_image_upload(file)
    state = _get_live_session(session_id[:80] or "default", min_hits)
    state["frame_index"] += 1
    frame_index = int(state["frame_index"])

    try:
        with _inference_lock:
            result = get_service().recognize_frame(
                frame,
                source_image="live",
                det_conf=det_conf,
                allow_edge=allow_edge,
                save_crops=False,
                save_annotated=False,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Live recognition failed: {type(exc).__name__}: {exc}") from exc

    temporal = state["tracker"].update(
        frame,
        result["results"],
        frame_index=frame_index,
        timestamp_s=time.monotonic(),
    )

    committed: list[dict[str, Any]] = []
    db_counts = {"inserted": 0, "duplicate": 0, "rejected": 0, "invalid": 0}
    candidates: list[TemporalTrack] = list(temporal["ready"])
    candidates.extend(track for track in temporal["finalized"] if not track.persisted and track.valid_hits >= 1)
    for track in candidates:
        if track.persisted:
            continue
        text, _ = track.consensus()
        if not text:
            continue
        if not _live_can_save(state, text):
            track.persisted = True
            continue
        persisted = persist_temporal_track(
            track,
            prefix="live_",
            source_type="live",
            source_ref=f"webcam:{session_id[:80]}",
        )
        track.persisted = True
        if persisted:
            committed.append(persisted)
            _sum_db_counts(db_counts, persisted["database"])

    committed_track_ids = {
        int(item["result"].get("track_id"))
        for item in committed
        if item.get("result", {}).get("track_id") is not None
    }
    response_results: list[dict[str, Any]] = []
    assignments: list[int | None] = temporal["assignments"]
    track_states: dict[int, dict[str, Any]] = temporal["tracks"]
    for index, item in enumerate(result["results"]):
        output = with_region(item)
        track_id = assignments[index] if index < len(assignments) else None
        snapshot = track_states.get(track_id) if track_id is not None else None
        output["track_id"] = track_id
        output["temporal_hits"] = int(snapshot.get("temporal_hits", 1)) if snapshot else 1
        output["temporal_valid_hits"] = int(snapshot.get("temporal_valid_hits", 0)) if snapshot else int(bool(item.get("format_valid")))
        output["temporal_confidence"] = float(snapshot.get("temporal_confidence", 0.0)) if snapshot else 0.0
        output["consensus_text"] = str(snapshot.get("raw_text", "")) if snapshot else ""
        output["live_persisted"] = track_id in committed_track_ids
        output["live_duplicate"] = False
        output["crop_url"] = None
        response_results.append(output)

    first_commit = committed[0] if committed else None
    return {
        "success": True,
        "mode": "live",
        "session_id": session_id,
        "event_id": first_commit.get("event_id") if first_commit else None,
        "original_url": first_commit.get("original_url") if first_commit else None,
        "annotated_url": first_commit.get("annotated_url") if first_commit else None,
        "detections": result["detections"],
        "accepted": result["accepted"],
        "rejected": result["rejected"],
        "database": db_counts,
        "dedup_seconds": LIVE_DEDUP_SECONDS,
        "active_tracks": len(state["tracker"].active_tracks),
        "committed": committed,
        "results": response_results,
    }


def _job_update(job_id: str, **values: Any) -> None:
    with _video_lock:
        job = _video_jobs.get(job_id)
        if job is not None:
            job.update(values)


def _job_append_event(job_id: str, event: dict[str, Any]) -> None:
    with _video_lock:
        job = _video_jobs.get(job_id)
        if job is not None:
            job.setdefault("events", []).append(event)
            job["saved_events"] = len(job["events"])


def _persist_video_track(
    job_id: str,
    track: TemporalTrack,
    video_relative: str,
    last_saved: dict[str, float],
) -> None:
    if track.persisted or track.valid_hits < 1:
        return
    text, _ = track.consensus()
    if not text:
        return
    timestamp_s = float(track.best_timestamp_s or 0.0)
    previous = last_saved.get(text)
    if previous is not None and abs(timestamp_s - previous) < VIDEO_DEDUP_SECONDS:
        track.persisted = True
        return
    last_saved[text] = timestamp_s
    persisted = persist_temporal_track(
        track,
        prefix="video_",
        source_type="video",
        source_ref=video_relative,
    )
    track.persisted = True
    if persisted:
        event = dict(persisted["result"])
        event["event_id"] = persisted["event_id"]
        event["annotated_url"] = persisted["annotated_url"]
        event["timestamp_seconds"] = timestamp_s
        _job_append_event(job_id, event)


def _process_video_job(
    job_id: str,
    video_path: Path,
    video_relative: str,
    det_conf: float,
    allow_edge: bool,
    sample_fps: float,
    min_hits: int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open the uploaded video.")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if total_frames > 0 else 0.0
        sample_every = max(1, int(round(fps / max(0.25, sample_fps))))
        estimated_samples = max(1, math.ceil(total_frames / sample_every)) if total_frames > 0 else 1

        _job_update(
            job_id,
            status="running",
            started_at=utc_now(),
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            sample_every=sample_every,
            estimated_samples=estimated_samples,
        )

        tracker = TemporalPlateTracker(min_hits=min_hits, max_missed_frames=3)
        last_saved: dict[str, float] = {}
        actual_frame = -1
        sample_index = 0
        total_detections = 0

        while True:
            ok = cap.grab()
            if not ok:
                break
            actual_frame += 1
            if actual_frame % sample_every != 0:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue

            sample_index += 1
            timestamp_s = actual_frame / fps
            with _inference_lock:
                result = get_service().recognize_frame(
                    frame,
                    source_image=video_relative,
                    det_conf=det_conf,
                    allow_edge=allow_edge,
                    save_crops=False,
                    save_annotated=False,
                )
            total_detections += int(result["detections"])
            temporal = tracker.update(
                frame,
                result["results"],
                frame_index=sample_index,
                timestamp_s=timestamp_s,
            )

            # For video we wait until a track ends so the saved evidence is the
            # sharpest frame observed across the whole vehicle pass.
            for track in temporal["finalized"]:
                _persist_video_track(job_id, track, video_relative, last_saved)

            progress = min(99.0, sample_index * 100.0 / max(1, estimated_samples))
            _job_update(
                job_id,
                progress=round(progress, 1),
                processed_samples=sample_index,
                detections=total_detections,
                current_time_seconds=timestamp_s,
            )

        for track in tracker.flush():
            _persist_video_track(job_id, track, video_relative, last_saved)

        with _video_lock:
            saved_events = len(_video_jobs.get(job_id, {}).get("events", []))
        _job_update(
            job_id,
            status="completed",
            progress=100.0,
            processed_samples=sample_index,
            detections=total_detections,
            saved_events=saved_events,
            completed_at=utc_now(),
        )
    except Exception as exc:
        _job_update(
            job_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            completed_at=utc_now(),
        )
    finally:
        cap.release()


@app.post("/api/recognize/video", status_code=202)
async def recognize_video(
    file: UploadFile = File(...),
    det_conf: float = Form(0.25, ge=0.01, le=0.99),
    allow_edge: bool = Form(False),
    sample_fps: float = Form(5.0, ge=0.25, le=15.0),
    min_hits: int = Form(2, ge=1, le=5),
):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}:
        suffix = ".mp4"
    job_id = new_event_id("job_")
    stored_name = f"{job_id}{suffix}"
    video_path = VIDEOS_DIR / stored_name
    video_relative = f"videos/{stored_name}"
    size_bytes = await save_video_upload(file, video_path)

    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": utc_now(),
        "started_at": None,
        "completed_at": None,
        "file_name": file.filename or stored_name,
        "video_url": media_url(video_relative),
        "size_bytes": size_bytes,
        "progress": 0.0,
        "sample_fps": sample_fps,
        "min_hits": min_hits,
        "processed_samples": 0,
        "estimated_samples": None,
        "detections": 0,
        "saved_events": 0,
        "events": [],
        "error": None,
    }
    with _video_lock:
        _video_jobs[job_id] = job

    thread = threading.Thread(
        target=_process_video_job,
        args=(job_id, video_path, video_relative, det_conf, allow_edge, sample_fps, min_hits),
        daemon=True,
        name=f"pelak-video-{job_id[-8:]}",
    )
    thread.start()
    return job


@app.get("/api/video/jobs")
def list_video_jobs(limit: int = Query(20, ge=1, le=100)):
    with _video_lock:
        jobs = list(_video_jobs.values())[-limit:]
        payload = [dict(job, events=list(job.get("events", []))) for job in reversed(jobs)]
    return {"count": len(payload), "items": payload}


@app.get("/api/video/jobs/{job_id}")
def get_video_job(job_id: str):
    with _video_lock:
        job = _video_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Video job not found")
        return dict(job, events=list(job.get("events", [])))


@app.get("/api/plate-region/{plate_text}")
def plate_region(plate_text: str):
    return lookup_plate_region(plate_text)


@app.get("/stats")
def stats():
    with connect(DB_PATH) as conn:
        unique_plates = conn.execute("SELECT COUNT(*) AS n FROM plates").fetchone()["n"]
        detections = conn.execute("SELECT COUNT(*) AS n FROM detections").fetchone()["n"]
        latest = conn.execute("SELECT MAX(detected_at) AS latest FROM detections").fetchone()["latest"]
        live_count = conn.execute("SELECT COUNT(*) AS n FROM detections WHERE source_type = 'live'").fetchone()["n"]
        video_count = conn.execute("SELECT COUNT(*) AS n FROM detections WHERE source_type = 'video'").fetchone()["n"]
    return {
        "unique_plates": unique_plates,
        "detection_events": detections,
        "latest_detection_at": latest,
        "live_events": live_count,
        "video_events": video_count,
    }


@app.get("/plates")
def list_plates(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    with connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT id, plate_text, display_text, first_seen_at, last_seen_at, detection_count
            FROM plates
            ORDER BY last_seen_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    items = [with_region(dict(row), "plate_text") for row in rows]
    return {"count": len(items), "limit": limit, "offset": offset, "items": items}


@app.get("/plates/{plate_text}")
def get_plate(plate_text: str):
    with connect(DB_PATH) as conn:
        plate = conn.execute(
            """
            SELECT id, plate_text, display_text, first_seen_at, last_seen_at, detection_count
            FROM plates WHERE plate_text = ?
            """,
            (plate_text,),
        ).fetchone()
        if plate is None:
            raise HTTPException(status_code=404, detail="Plate not found")

        detections = conn.execute(
            """
            SELECT id, detected_at, source_image, plate_index, det_confidence,
                   x1, y1, x2, y2, crop_path, edge_suspect, raw_text, display_text,
                   source_type, source_ref, source_time_seconds, track_id, sharpness,
                   temporal_hits, temporal_confidence
            FROM detections
            WHERE plate_id = ?
            ORDER BY detected_at DESC, id DESC
            """,
            (plate["id"],),
        ).fetchall()

    detection_items = rows_to_dicts(detections)
    for item in detection_items:
        item["source_url"] = media_url(item.get("source_image"))
        item["crop_url"] = media_url(item.get("crop_path"))
        item["region"] = lookup_plate_region(str(item.get("raw_text", "")))
    return {"plate": with_region(dict(plate), "plate_text"), "detections": detection_items}


@app.get("/detections")
def list_detections(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    with connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT d.id, p.plate_text, p.display_text, d.detected_at,
                   d.source_image, d.plate_index, d.det_confidence,
                   d.x1, d.y1, d.x2, d.y2, d.crop_path, d.edge_suspect,
                   d.source_type, d.source_ref, d.source_time_seconds, d.track_id,
                   d.sharpness, d.temporal_hits, d.temporal_confidence
            FROM detections d
            JOIN plates p ON p.id = d.plate_id
            ORDER BY d.detected_at DESC, d.id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    items = rows_to_dicts(rows)
    for item in items:
        item["source_url"] = media_url(item.get("source_image"))
        item["crop_url"] = media_url(item.get("crop_path"))
        item["region"] = lookup_plate_region(str(item.get("plate_text", "")))
    return {"count": len(items), "limit": limit, "offset": offset, "items": items}


@app.get("/search")
def search(q: str = Query(..., min_length=1, max_length=64), limit: int = Query(50, ge=1, le=500)):
    pattern = f"%{q.strip()}%"
    with connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT id, plate_text, display_text, first_seen_at, last_seen_at, detection_count
            FROM plates
            WHERE plate_text LIKE ? OR display_text LIKE ?
            ORDER BY last_seen_at DESC
            LIMIT ?
            """,
            (pattern, pattern, limit),
        ).fetchall()
    items = [with_region(dict(row), "plate_text") for row in rows]
    return {"query": q, "count": len(items), "items": items}


@app.delete("/detections/{detection_id}")
def remove_detection(detection_id: int, delete_media: bool = Query(True)):
    deleted = delete_detection(DB_PATH, detection_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    media_removed = 0
    if delete_media:
        media_removed = _cleanup_detection_media(
            str(deleted.get("source_image") or ""),
            str(deleted.get("crop_path") or ""),
        )
    return {"success": True, "detection_id": detection_id, "plate_removed": deleted["plate_removed"], "media_removed": media_removed}


@app.delete("/plates/{plate_text}")
def remove_plate(plate_text: str, delete_media: bool = Query(True)):
    deleted = delete_plate(DB_PATH, plate_text)
    if deleted is None:
        raise HTTPException(status_code=404, detail="Plate not found")
    media_removed = 0
    if delete_media:
        for item in deleted["media"]:
            media_removed += _cleanup_detection_media(
                str(item.get("source_image") or ""),
                str(item.get("crop_path") or ""),
            )
    return {"success": True, "plate_text": plate_text, "deleted_detections": deleted["deleted_detections"], "media_removed": media_removed}


@app.get("/api/database/backup")
def backup_database():
    filename = f"pelak_khan_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    destination = BACKUPS_DIR / filename
    create_backup(DB_PATH, destination)
    return FileResponse(destination, filename=filename, media_type="application/octet-stream")


@app.delete("/api/database")
def remove_all_database(
    confirm: str = Query(...),
    delete_media: bool = Query(True),
):
    if confirm != "DELETE_ALL":
        raise HTTPException(status_code=400, detail="Confirmation token is invalid")

    media_rows: list[dict[str, Any]] = []
    if delete_media:
        with connect(DB_PATH) as conn:
            media_rows = rows_to_dicts(conn.execute("SELECT source_image, crop_path FROM detections").fetchall())
    counts = clear_database(DB_PATH)
    removed = 0
    if delete_media:
        seen: set[tuple[str, str]] = set()
        for row in media_rows:
            pair = (str(row.get("source_image") or ""), str(row.get("crop_path") or ""))
            if pair in seen:
                continue
            seen.add(pair)
            removed += _cleanup_detection_media(*pair)
    return {"success": True, "deleted": counts, "media_removed": removed}
