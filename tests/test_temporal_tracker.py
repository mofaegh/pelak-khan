import numpy as np

from pelak_khan.tracking.temporal import TemporalPlateTracker


def rec(text, x, conf=0.9, sharp=100.0):
    return {
        "raw_text": text,
        "display_text": text,
        "format_valid": len(text) == 8,
        "accepted": len(text) == 8,
        "det_confidence": conf,
        "sharpness": sharp,
        "x1": x,
        "y1": 100,
        "x2": x + 120,
        "y2": 140,
    }


def test_temporal_vote_and_motion_tracking():
    tracker = TemporalPlateTracker(min_hits=2, max_missed_frames=2)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    a = tracker.update(frame, [rec("18ق26744", 100, sharp=80)], frame_index=1, timestamp_s=0.0)
    assert not a["ready"]

    b = tracker.update(frame, [rec("18ق26745", 220, sharp=160)], frame_index=2, timestamp_s=0.2)
    assert len(b["ready"]) == 1
    track = b["ready"][0]
    voted, confidence = track.consensus()
    assert len(voted) == 8
    assert voted.startswith("18ق2674")
    assert track.hits == 2
    assert track.best_record["sharpness"] == 160
    assert confidence > 0


def test_one_frame_track_finalizes_for_fast_vehicle():
    tracker = TemporalPlateTracker(min_hits=2, max_missed_frames=1)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker.update(frame, [rec("18ق26744", 100)], frame_index=1, timestamp_s=0.0)
    tracker.update(frame, [], frame_index=2, timestamp_s=0.2)
    result = tracker.update(frame, [], frame_index=3, timestamp_s=0.4)
    assert len(result["finalized"]) == 1
    assert result["finalized"][0].valid_hits == 1
