"""ByteTrack speed-ups must not change what the tracker outputs.

The reference was recorded with the tracker before the speed-up (fork 0dc53f92) by
make_bytetrack_reference.py. Every detection must get the same tracker id (same
tracks, same order of creation, same crossings downstream). With
BYTETRACK_REFERENCE_DIR set, full-length recordings are also checked, including the
Kalman state of every output track (rounding only: <= 1e-4 px).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from test.tracker.bytetrack_replay import CONFIGS, replay

REFERENCE = Path(__file__).parent / "data" / "bytetrack_reference.npz"
FULL_DIR = os.environ.get("BYTETRACK_REFERENCE_DIR")
STATE_ATOL = 1e-4


def _sequences():
    with np.load(REFERENCE) as z:
        return sorted({key.split("/")[0] for key in z.files})


@pytest.mark.parametrize("config", sorted(CONFIGS))
@pytest.mark.parametrize("sequence", _sequences())
def test_same_tracker_ids_as_reference(sequence: str, config: str) -> None:
    with np.load(REFERENCE) as z:
        tracker_ids = replay(
            z[f"{sequence}/frame"],
            z[f"{sequence}/xyxy"],
            z[f"{sequence}/conf"],
            int(z[f"{sequence}/n_frames"]),
            CONFIGS[config],
        )
        expected = z[f"{sequence}/{config}/tracker_id"]
    differ = np.flatnonzero(tracker_ids != expected)
    assert len(differ) == 0, (
        f"{len(differ)} detections with another tracker id; first at index "
        f"{differ[0]}: {tracker_ids[differ[0]]} instead of {expected[differ[0]]}"
    )


def _full_recordings():
    if not FULL_DIR:
        return []
    return sorted(
        p for p in Path(FULL_DIR).glob("*.npz") if not p.name.endswith(".ref.npz")
    )


@pytest.mark.skipif(not FULL_DIR, reason="BYTETRACK_REFERENCE_DIR not set")
@pytest.mark.parametrize("config", sorted(CONFIGS))
@pytest.mark.parametrize("recording", _full_recordings(), ids=lambda p: p.stem)
def test_full_recording_same_ids_and_states(recording: Path, config: str) -> None:
    z = np.load(recording)
    ref = np.load(recording.with_name(recording.stem + ".ref.npz"))
    tracker_ids, state_frame, state_id, state_tlbr = replay(
        z["frame"],
        z["xyxy"],
        z["conf"],
        int(z["n_frames"]),
        CONFIGS[config],
        with_states=True,
    )
    np.testing.assert_array_equal(tracker_ids, ref[f"{config}/tracker_id"])
    np.testing.assert_array_equal(state_frame, ref[f"{config}/state_frame"])
    np.testing.assert_array_equal(state_id, ref[f"{config}/state_id"])
    np.testing.assert_allclose(
        state_tlbr, ref[f"{config}/state_tlbr"], rtol=0, atol=STATE_ATOL
    )
