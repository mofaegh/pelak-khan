#!/usr/bin/env python
"""
Pelak-Khan SQLite persistence for validated ALPR results.

Input:
    results.json produced by scripts/12_alpr_image_validated.py

Creates/updates:
    data/pelak_khan.db   (default)

Schema:
    plates
      - normalized plate identity and aggregate counters
    detections
      - every accepted detection event with source image, confidence, bbox, crop, time

Behavior:
    - only accepted=True rows are inserted
    - re-ingesting the same result is idempotent via event_key UNIQUE
    - plate aggregate counters are updated automatically
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    FOREIGN KEY (plate_id) REFERENCES plates(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_detections_plate_id
ON detections(plate_id);

CREATE INDEX IF NOT EXISTS idx_detections_detected_at
ON detections(detected_at);

CREATE INDEX IF NOT EXISTS idx_detections_source_image
ON detections(source_image);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_event_key(rec: dict[str, Any]) -> str:
    parts = [
        str(rec.get("source_image", "")),
        str(rec.get("plate_index", "")),
        str(rec.get("raw_text", "")),
        str(rec.get("x1", "")),
        str(rec.get("y1", "")),
        str(rec.get("x2", "")),
        str(rec.get("y2", "")),
        str(rec.get("crop_path", "")),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def get_or_create_plate(
    conn: sqlite3.Connection,
    plate_text: str,
    display_text: str,
    seen_at: str,
) -> tuple[int, bool]:
    row = conn.execute(
        "SELECT id FROM plates WHERE plate_text = ?",
        (plate_text,),
    ).fetchone()

    if row:
        return int(row[0]), False

    cur = conn.execute(
        """
        INSERT INTO plates (
            plate_text, display_text, first_seen_at, last_seen_at, detection_count
        )
        VALUES (?, ?, ?, ?, 0)
        """,
        (plate_text, display_text, seen_at, seen_at),
    )
    return int(cur.lastrowid), True


def ingest_one(
    conn: sqlite3.Connection,
    rec: dict[str, Any],
    detected_at: str,
) -> str:
    if not bool(rec.get("accepted", False)):
        return "rejected"

    plate_text = str(rec.get("raw_text", "")).strip()
    display_text = str(rec.get("display_text", plate_text)).strip()

    if not plate_text:
        return "invalid"

    event_key = make_event_key(rec)

    exists = conn.execute(
        "SELECT 1 FROM detections WHERE event_key = ?",
        (event_key,),
    ).fetchone()
    if exists:
        return "duplicate"

    plate_id, _ = get_or_create_plate(
        conn,
        plate_text,
        display_text,
        detected_at,
    )

    conn.execute(
        """
        INSERT INTO detections (
            plate_id,
            event_key,
            detected_at,
            source_image,
            plate_index,
            det_confidence,
            x1, y1, x2, y2,
            crop_path,
            edge_suspect,
            raw_text,
            display_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            plate_id,
            event_key,
            detected_at,
            str(rec.get("source_image", "")),
            rec.get("plate_index"),
            rec.get("det_confidence"),
            rec.get("x1"),
            rec.get("y1"),
            rec.get("x2"),
            rec.get("y2"),
            str(rec.get("crop_path", "")),
            1 if bool(rec.get("edge_suspect", False)) else 0,
            plate_text,
            display_text,
        ),
    )

    conn.execute(
        """
        UPDATE plates
        SET last_seen_at = ?,
            detection_count = detection_count + 1,
            display_text = ?
        WHERE id = ?
        """,
        (detected_at, display_text, plate_id),
    )

    return "inserted"


def print_recent(conn: sqlite3.Connection, limit: int = 10) -> None:
    rows = conn.execute(
        """
        SELECT
            d.id,
            p.plate_text,
            p.display_text,
            d.det_confidence,
            d.detected_at,
            d.source_image
        FROM detections d
        JOIN plates p ON p.id = d.plate_id
        ORDER BY d.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    print()
    print("Recent accepted detections")
    print("-" * 100)
    if not rows:
        print("(none)")
        return

    for row in rows:
        det_id, plate, display, conf, detected_at, source = row
        conf_text = "n/a" if conf is None else f"{float(conf):.3f}"
        print(
            f"id={det_id:<4} plate={plate:<12} display={display:<16} "
            f"det={conf_text:<6} at={detected_at} source={source}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="results.json produced by scripts/12_alpr_image_validated.py",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite DB path. Default: <repo>/data/pelak_khan.db",
    )
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    db_path = args.db or (repo_root / "data" / "pelak_khan.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.results.exists():
        print(f"ERROR: results file not found: {args.results}", file=sys.stderr)
        return 2

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    records = payload.get("results", [])
    if not isinstance(records, list):
        print("ERROR: results.json has no valid 'results' list.", file=sys.stderr)
        return 2

    # For now one ingestion timestamp is used for the batch.
    # Later camera/video mode will pass frame/capture timestamps per event.
    detected_at = utc_now()

    counts = {
        "inserted": 0,
        "duplicate": 0,
        "rejected": 0,
        "invalid": 0,
    }

    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)

        with conn:
            for rec in records:
                status = ingest_one(conn, rec, detected_at)
                counts[status] = counts.get(status, 0) + 1

        plate_total = conn.execute(
            "SELECT COUNT(*) FROM plates"
        ).fetchone()[0]
        detection_total = conn.execute(
            "SELECT COUNT(*) FROM detections"
        ).fetchone()[0]

        print("=" * 100)
        print("Pelak-Khan SQLite Ingest Complete")
        print("=" * 100)
        print(f"Input results     : {args.results}")
        print(f"Database          : {db_path}")
        print(f"Inserted accepted : {counts['inserted']}")
        print(f"Duplicate skipped : {counts['duplicate']}")
        print(f"Rejected skipped  : {counts['rejected']}")
        print(f"Invalid skipped   : {counts['invalid']}")
        print(f"Unique plates DB  : {plate_total}")
        print(f"Detection events  : {detection_total}")
        print("=" * 100)

        print_recent(conn, args.show)

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
