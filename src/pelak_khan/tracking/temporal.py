from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import hypot
from typing import Any

import numpy as np


def _box(record: dict[str, Any]) -> tuple[float, float, float, float]:
    return tuple(float(record.get(k, 0.0)) for k in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_distance_norm(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    diagonal = max(1.0, hypot(frame_width, frame_height))
    return hypot(acx - bcx, acy - bcy) / diagonal


def _valid_text(record: dict[str, Any]) -> str | None:
    text = str(record.get("raw_text", "")).strip()
    if bool(record.get("format_valid")) and len(text) == 8:
        return text
    return None


def _char_vote(texts: list[str]) -> tuple[str, float]:
    if not texts:
        return "", 0.0
    if len(texts) == 1:
        return texts[0], 1.0

    voted: list[str] = []
    supports: list[float] = []
    for position in range(8):
        counts = Counter(text[position] for text in texts)
        char, count = counts.most_common(1)[0]
        voted.append(char)
        supports.append(count / len(texts))
    return "".join(voted), sum(supports) / len(supports)


@dataclass
class TemporalTrack:
    track_id: int
    created_frame: int
    last_frame: int
    last_box: tuple[float, float, float, float]
    observations: list[dict[str, Any]] = field(default_factory=list)
    best_record: dict[str, Any] | None = None
    best_frame: np.ndarray | None = None
    best_score: float = -1.0
    best_timestamp_s: float = 0.0
    persisted: bool = False
    ready_emitted: bool = False

    def add(self, frame: np.ndarray, record: dict[str, Any], frame_index: int, timestamp_s: float) -> None:
        copied = dict(record)
        copied["frame_index"] = frame_index
        copied["timestamp_s"] = timestamp_s
        self.observations.append(copied)
        self.last_frame = frame_index
        self.last_box = _box(record)

        sharpness = float(record.get("sharpness", 0.0) or 0.0)
        det_conf = float(record.get("det_confidence", 0.0) or 0.0)
        valid_bonus = 1.35 if _valid_text(record) else 1.0
        score = (max(1.0, sharpness) ** 0.5) * (0.5 + det_conf) * valid_bonus
        if self.best_record is None or score > self.best_score:
            self.best_score = score
            self.best_record = copied
            self.best_frame = frame.copy()
            self.best_timestamp_s = timestamp_s

    @property
    def hits(self) -> int:
        return len(self.observations)

    @property
    def valid_texts(self) -> list[str]:
        return [text for record in self.observations if (text := _valid_text(record))]

    @property
    def valid_hits(self) -> int:
        return len(self.valid_texts)

    def consensus(self) -> tuple[str, float]:
        texts = self.valid_texts
        if not texts:
            return "", 0.0

        exact = Counter(texts)
        exact_text, exact_count = exact.most_common(1)[0]
        if exact_count >= 2:
            return exact_text, exact_count / len(texts)

        return _char_vote(texts)

    def snapshot(self, reason: str = "active") -> dict[str, Any]:
        text, confidence = self.consensus()
        record = dict(self.best_record or (self.observations[-1] if self.observations else {}))
        if text:
            record["raw_text"] = text
            record["format_valid"] = True
            record["accepted"] = True
            record["status"] = "ACCEPTED_TEMPORAL"
            record["reject_reasons"] = []
        record.update(
            {
                "track_id": self.track_id,
                "temporal_hits": self.hits,
                "temporal_valid_hits": self.valid_hits,
                "temporal_confidence": confidence,
                "track_reason": reason,
                "source_time_seconds": self.best_timestamp_s,
            }
        )
        return record


class TemporalPlateTracker:
    """Small multi-object tracker + OCR temporal voter for ALPR.

    Association is deliberately lightweight: IoU plus normalized centre distance.
    This keeps the runtime dependency-free and works well for fixed parking/CCTV
    cameras where plates travel predictably through the frame.
    """

    def __init__(
        self,
        *,
        min_hits: int = 2,
        max_missed_frames: int = 3,
        max_center_distance: float = 0.28,
        min_iou: float = 0.03,
    ) -> None:
        self.min_hits = max(1, int(min_hits))
        self.max_missed_frames = max(1, int(max_missed_frames))
        self.max_center_distance = float(max_center_distance)
        self.min_iou = float(min_iou)
        self._next_id = 1
        self._tracks: dict[int, TemporalTrack] = {}

    @property
    def active_tracks(self) -> list[TemporalTrack]:
        return list(self._tracks.values())

    def _match_score(
        self,
        track: TemporalTrack,
        record: dict[str, Any],
        frame_width: int,
        frame_height: int,
    ) -> float | None:
        box = _box(record)
        overlap = _iou(track.last_box, box)
        distance = _center_distance_norm(track.last_box, box, frame_width, frame_height)
        if overlap < self.min_iou and distance > self.max_center_distance:
            return None
        distance_score = max(0.0, 1.0 - distance / max(self.max_center_distance, 1e-6))
        return overlap * 2.0 + distance_score

    def update(
        self,
        frame: np.ndarray,
        records: list[dict[str, Any]],
        *,
        frame_index: int,
        timestamp_s: float,
    ) -> dict[str, Any]:
        height, width = frame.shape[:2]
        unmatched_tracks = set(self._tracks)
        assignments: list[int | None] = [None] * len(records)

        # High-confidence detections claim tracks first.
        order = sorted(
            range(len(records)),
            key=lambda i: float(records[i].get("det_confidence", 0.0) or 0.0),
            reverse=True,
        )
        for record_index in order:
            record = records[record_index]
            candidates: list[tuple[float, int]] = []
            for track_id in unmatched_tracks:
                score = self._match_score(self._tracks[track_id], record, width, height)
                if score is not None:
                    candidates.append((score, track_id))

            if candidates:
                _, track_id = max(candidates)
                track = self._tracks[track_id]
                unmatched_tracks.discard(track_id)
            else:
                track_id = self._next_id
                self._next_id += 1
                track = TemporalTrack(
                    track_id=track_id,
                    created_frame=frame_index,
                    last_frame=frame_index,
                    last_box=_box(record),
                )
                self._tracks[track_id] = track

            track.add(frame, record, frame_index, timestamp_s)
            assignments[record_index] = track_id

        ready: list[TemporalTrack] = []
        for track in self._tracks.values():
            if not track.ready_emitted and track.valid_hits >= self.min_hits:
                text, _ = track.consensus()
                if text:
                    track.ready_emitted = True
                    ready.append(track)

        finalized: list[TemporalTrack] = []
        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame > self.max_missed_frames
        ]
        for track_id in stale_ids:
            finalized.append(self._tracks.pop(track_id))

        track_states = {track.track_id: track.snapshot() for track in self._tracks.values()}
        return {
            "assignments": assignments,
            "ready": ready,
            "finalized": finalized,
            "tracks": track_states,
        }

    def flush(self) -> list[TemporalTrack]:
        tracks = list(self._tracks.values())
        self._tracks.clear()
        return tracks
