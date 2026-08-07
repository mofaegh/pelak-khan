#!/usr/bin/env python
"""
Pelak-Khan FastAPI backend v1.

Reads the SQLite database created by scripts/13_ingest_alpr_sqlite.py.

Endpoints:
    GET /health
    GET /stats
    GET /plates
    GET /plates/{plate_text}
    GET /detections
    GET /search?q=...

Run:
    uvicorn apps.backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "pelak_khan.db"
DB_PATH = Path(os.getenv("PELAK_DB_PATH", str(DEFAULT_DB_PATH)))


app = FastAPI(
    title="Pelak-Khan API",
    version="0.1.0",
    description="Iranian license plate recognition backend",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Database not found: {DB_PATH}",
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def rows_to_dicts(rows) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "database": str(DB_PATH),
        "database_exists": DB_PATH.exists(),
    }


@app.get("/stats")
def stats():
    with get_conn() as conn:
        unique_plates = conn.execute(
            "SELECT COUNT(*) AS n FROM plates"
        ).fetchone()["n"]

        detections = conn.execute(
            "SELECT COUNT(*) AS n FROM detections"
        ).fetchone()["n"]

        latest = conn.execute(
            "SELECT MAX(detected_at) AS latest FROM detections"
        ).fetchone()["latest"]

    return {
        "unique_plates": unique_plates,
        "detection_events": detections,
        "latest_detection_at": latest,
    }


@app.get("/plates")
def list_plates(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                plate_text,
                display_text,
                first_seen_at,
                last_seen_at,
                detection_count
            FROM plates
            ORDER BY last_seen_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    return {
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "items": rows_to_dicts(rows),
    }


@app.get("/plates/{plate_text}")
def get_plate(plate_text: str):
    with get_conn() as conn:
        plate = conn.execute(
            """
            SELECT
                id,
                plate_text,
                display_text,
                first_seen_at,
                last_seen_at,
                detection_count
            FROM plates
            WHERE plate_text = ?
            """,
            (plate_text,),
        ).fetchone()

        if plate is None:
            raise HTTPException(status_code=404, detail="Plate not found")

        detections = conn.execute(
            """
            SELECT
                id,
                detected_at,
                source_image,
                plate_index,
                det_confidence,
                x1, y1, x2, y2,
                crop_path,
                edge_suspect,
                raw_text,
                display_text
            FROM detections
            WHERE plate_id = ?
            ORDER BY detected_at DESC, id DESC
            """,
            (plate["id"],),
        ).fetchall()

    return {
        "plate": dict(plate),
        "detections": rows_to_dicts(detections),
    }


@app.get("/detections")
def list_detections(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                d.id,
                p.plate_text,
                p.display_text,
                d.detected_at,
                d.source_image,
                d.plate_index,
                d.det_confidence,
                d.x1, d.y1, d.x2, d.y2,
                d.crop_path,
                d.edge_suspect
            FROM detections d
            JOIN plates p ON p.id = d.plate_id
            ORDER BY d.detected_at DESC, d.id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    return {
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "items": rows_to_dicts(rows),
    }


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, max_length=64),
    limit: int = Query(50, ge=1, le=500),
):
    pattern = f"%{q.strip()}%"

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                plate_text,
                display_text,
                first_seen_at,
                last_seen_at,
                detection_count
            FROM plates
            WHERE plate_text LIKE ?
               OR display_text LIKE ?
            ORDER BY last_seen_at DESC
            LIMIT ?
            """,
            (pattern, pattern, limit),
        ).fetchall()

    return {
        "query": q,
        "count": len(rows),
        "items": rows_to_dicts(rows),
    }
