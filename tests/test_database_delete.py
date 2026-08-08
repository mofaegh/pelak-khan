from pathlib import Path

from pelak_khan.storage.database import connect, delete_detection, delete_plate, ingest_records


def record(text, source, idx):
    return {
        "accepted": True,
        "raw_text": text,
        "display_text": text,
        "source_image": source,
        "crop_path": f"crops/{idx}.jpg",
        "plate_index": idx,
        "det_confidence": 0.9,
        "x1": 1,
        "y1": 2,
        "x2": 3,
        "y2": 4,
        "edge_suspect": False,
        "source_type": "image",
    }


def test_delete_detection_refreshes_plate(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    ingest_records(db, [record("18ق26744", "originals/a.jpg", 1)])
    with connect(db) as conn:
        det_id = conn.execute("SELECT id FROM detections").fetchone()["id"]
    deleted = delete_detection(db, det_id)
    assert deleted is not None
    assert deleted["plate_removed"] is True
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM plates").fetchone()["n"] == 0


def test_delete_whole_plate(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    ingest_records(db, [record("18ق26744", "originals/a.jpg", 1), record("18ق26744", "originals/b.jpg", 2)])
    deleted = delete_plate(db, "18ق26744")
    assert deleted["deleted_detections"] == 2


def test_migrates_existing_v02_database(tmp_path: Path):
    import sqlite3
    db = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("""
    CREATE TABLE plates (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      plate_text TEXT NOT NULL UNIQUE,
      display_text TEXT NOT NULL,
      first_seen_at TEXT NOT NULL,
      last_seen_at TEXT NOT NULL,
      detection_count INTEGER NOT NULL DEFAULT 1
    );
    CREATE TABLE detections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      plate_id INTEGER NOT NULL,
      event_key TEXT NOT NULL UNIQUE,
      detected_at TEXT NOT NULL,
      source_image TEXT NOT NULL,
      plate_index INTEGER,
      det_confidence REAL,
      x1 REAL, y1 REAL, x2 REAL, y2 REAL,
      crop_path TEXT,
      edge_suspect INTEGER NOT NULL DEFAULT 0,
      raw_text TEXT NOT NULL,
      display_text TEXT NOT NULL
    );
    """)
    conn.close()
    with connect(db) as migrated:
        cols = {row["name"] for row in migrated.execute("PRAGMA table_info(detections)").fetchall()}
    assert "source_type" in cols
    assert "temporal_confidence" in cols
