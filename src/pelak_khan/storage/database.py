from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS plates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_text TEXT NOT NULL UNIQUE,
    display_text TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    detection_count INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_id INTEGER NOT NULL,
    event_key TEXT NOT NULL UNIQUE,
    detected_at TEXT NOT NULL,
    source_image TEXT NOT NULL,
    plate_index INTEGER,
    det_confidence REAL,
    x1 REAL,
    y1 REAL,
    x2 REAL,
    y2 REAL,
    crop_path TEXT,
    edge_suspect INTEGER NOT NULL DEFAULT 0,
    raw_text TEXT NOT NULL,
    display_text TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'image',
    source_ref TEXT,
    source_time_seconds REAL,
    track_id TEXT,
    sharpness REAL,
    temporal_hits INTEGER,
    temporal_confidence REAL,
    FOREIGN KEY (plate_id) REFERENCES plates(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_detections_plate_id ON detections(plate_id);
CREATE INDEX IF NOT EXISTS idx_detections_detected_at ON detections(detected_at);
CREATE INDEX IF NOT EXISTS idx_detections_source_image ON detections(source_image);
"""

EXTRA_DETECTION_COLUMNS: dict[str, str] = {
    "source_type": "TEXT NOT NULL DEFAULT 'image'",
    "source_ref": "TEXT",
    "source_time_seconds": "REAL",
    "track_id": "TEXT",
    "sharpness": "REAL",
    "temporal_hits": "INTEGER",
    "temporal_confidence": "REAL",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _migrate(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(detections)").fetchall()}
    for name, sql_type in EXTRA_DETECTION_COLUMNS.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE detections ADD COLUMN {name} {sql_type}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_source_type ON detections(source_type)")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def make_event_key(record: dict[str, Any]) -> str:
    parts = [
        str(record.get("source_image", "")),
        str(record.get("plate_index", "")),
        str(record.get("raw_text", "")),
        str(record.get("x1", "")),
        str(record.get("y1", "")),
        str(record.get("x2", "")),
        str(record.get("y2", "")),
        str(record.get("crop_path", "")),
        str(record.get("source_type", "")),
        str(record.get("source_ref", "")),
        str(record.get("source_time_seconds", "")),
        str(record.get("track_id", "")),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _get_or_create_plate(
    conn: sqlite3.Connection,
    plate_text: str,
    display_text: str,
    seen_at: str,
) -> int:
    row = conn.execute("SELECT id FROM plates WHERE plate_text = ?", (plate_text,)).fetchone()
    if row:
        return int(row["id"])

    cursor = conn.execute(
        """
        INSERT INTO plates (plate_text, display_text, first_seen_at, last_seen_at, detection_count)
        VALUES (?, ?, ?, ?, 0)
        """,
        (plate_text, display_text, seen_at, seen_at),
    )
    return int(cursor.lastrowid)


def ingest_one(conn: sqlite3.Connection, record: dict[str, Any], detected_at: str) -> str:
    if not bool(record.get("accepted", False)):
        return "rejected"

    plate_text = str(record.get("raw_text", "")).strip()
    display_text = str(record.get("display_text", plate_text)).strip()
    if not plate_text:
        return "invalid"

    event_key = make_event_key(record)
    if conn.execute("SELECT 1 FROM detections WHERE event_key = ?", (event_key,)).fetchone():
        return "duplicate"

    plate_id = _get_or_create_plate(conn, plate_text, display_text, detected_at)

    conn.execute(
        """
        INSERT INTO detections (
            plate_id, event_key, detected_at, source_image, plate_index,
            det_confidence, x1, y1, x2, y2, crop_path, edge_suspect,
            raw_text, display_text, source_type, source_ref, source_time_seconds,
            track_id, sharpness, temporal_hits, temporal_confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            plate_id,
            event_key,
            detected_at,
            str(record.get("source_image", "")),
            record.get("plate_index"),
            record.get("det_confidence"),
            record.get("x1"),
            record.get("y1"),
            record.get("x2"),
            record.get("y2"),
            str(record.get("crop_path", "")),
            1 if bool(record.get("edge_suspect", False)) else 0,
            plate_text,
            display_text,
            str(record.get("source_type", "image") or "image"),
            str(record.get("source_ref", "") or ""),
            record.get("source_time_seconds"),
            str(record.get("track_id", "") or ""),
            record.get("sharpness"),
            record.get("temporal_hits"),
            record.get("temporal_confidence"),
        ),
    )

    conn.execute(
        """
        UPDATE plates
        SET last_seen_at = ?, detection_count = detection_count + 1, display_text = ?
        WHERE id = ?
        """,
        (detected_at, display_text, plate_id),
    )
    return "inserted"


def ingest_records(
    db_path: Path,
    records: Iterable[dict[str, Any]],
    detected_at: str | None = None,
) -> dict[str, int]:
    timestamp = detected_at or utc_now()
    counts = {"inserted": 0, "duplicate": 0, "rejected": 0, "invalid": 0}
    with connect(db_path) as conn:
        for record in records:
            status = ingest_one(conn, record, timestamp)
            counts[status] = counts.get(status, 0) + 1
    return counts


def _refresh_plate(conn: sqlite3.Connection, plate_id: int) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*) AS n, MIN(detected_at) AS first_seen, MAX(detected_at) AS last_seen
        FROM detections WHERE plate_id = ?
        """,
        (plate_id,),
    ).fetchone()
    count = int(row["n"] or 0)
    if count == 0:
        conn.execute("DELETE FROM plates WHERE id = ?", (plate_id,))
        return True
    conn.execute(
        """
        UPDATE plates SET detection_count = ?, first_seen_at = ?, last_seen_at = ? WHERE id = ?
        """,
        (count, row["first_seen"], row["last_seen"], plate_id),
    )
    return False


def delete_detection(db_path: Path, detection_id: int) -> dict[str, Any] | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, plate_id, source_image, crop_path, source_ref FROM detections WHERE id = ?",
            (detection_id,),
        ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        conn.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
        payload["plate_removed"] = _refresh_plate(conn, int(row["plate_id"]))
        return payload


def delete_plate(db_path: Path, plate_text: str) -> dict[str, Any] | None:
    with connect(db_path) as conn:
        plate = conn.execute("SELECT id, plate_text FROM plates WHERE plate_text = ?", (plate_text,)).fetchone()
        if plate is None:
            return None
        rows = conn.execute(
            "SELECT id, source_image, crop_path, source_ref FROM detections WHERE plate_id = ?",
            (plate["id"],),
        ).fetchall()
        conn.execute("DELETE FROM plates WHERE id = ?", (plate["id"],))
        return {"plate_text": plate_text, "deleted_detections": len(rows), "media": [dict(r) for r in rows]}


def clear_database(db_path: Path) -> dict[str, int]:
    with connect(db_path) as conn:
        detections = int(conn.execute("SELECT COUNT(*) AS n FROM detections").fetchone()["n"])
        plates = int(conn.execute("SELECT COUNT(*) AS n FROM plates").fetchone()["n"])
        conn.execute("DELETE FROM detections")
        conn.execute("DELETE FROM plates")
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('plates', 'detections')")
    return {"plates": plates, "detections": detections}


def is_media_referenced(db_path: Path, relative_path: str) -> bool:
    if not relative_path:
        return False
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM detections WHERE source_image = ? OR crop_path = ? OR source_ref = ? LIMIT 1",
            (relative_path, relative_path, relative_path),
        ).fetchone()
    return row is not None


def create_backup(db_path: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as source:
        target = sqlite3.connect(destination)
        try:
            source.backup(target)
        finally:
            target.close()
    return destination
