"""Replay of recorded or synthetic detections through ByteTrack, frame by frame.

Shared by the equivalence test and by make_bytetrack_reference.py, which records the
reference outputs with the tracker as it was before the speed-up (fork 0dc53f92).
"""

from __future__ import annotations

import numpy as np

import supervision as sv

# Tracker configurations covered by the reference (Ball Counter v2 uses frame_rate=60)
CONFIGS = {
    "base": {},
    "prior": {"velocity_prior_alpha": 0.1},
    "mcf3": {"minimum_consecutive_frames": 3},
    "prior_mcf3": {"velocity_prior_alpha": 0.1, "minimum_consecutive_frames": 3},
}
COMMON = {
    "track_activation_threshold": 0.25,
    "lost_track_buffer": 30,
    "minimum_matching_threshold": 0.8,
    "frame_rate": 60,
}


def replay(frame, xyxy, conf, n_frames, params, with_states=False):
    """Run ByteTrack over detections grouped by `frame` (1-based, sorted).

    Returns the tracker id given to every input detection (-1 when the tracker
    drops it) and, with `with_states`, the output tracks of every frame
    (frame, external id, Kalman tlbr).
    """
    tracker = sv.ByteTrack(**COMMON, **params)
    if with_states:
        captured = {}
        update_with_tensors = tracker.update_with_tensors

        def capture(tensors):
            captured["tracks"] = update_with_tensors(tensors)
            return captured["tracks"]

        tracker.update_with_tensors = capture

    bounds = np.searchsorted(frame, np.arange(1, n_frames + 2))
    tracker_ids = np.full(len(frame), -1, dtype=np.int32)
    state_frame, state_id, state_tlbr = [], [], []
    for f in range(1, n_frames + 1):
        a, b = bounds[f - 1], bounds[f]
        detections = sv.Detections(
            xyxy=xyxy[a:b],
            confidence=conf[a:b],
            class_id=np.zeros(b - a, dtype=int),
            data={"index": np.arange(a, b)},
        )
        out = tracker.update_with_detections(detections)
        if len(out):
            tracker_ids[out.data["index"]] = out.tracker_id
        if with_states:
            for track in captured.pop("tracks", []):
                state_frame.append(f)
                state_id.append(track.external_track_id)
                state_tlbr.append(track.tlbr)
    if not with_states:
        return tracker_ids
    return (
        tracker_ids,
        np.asarray(state_frame, dtype=np.int32),
        np.asarray(state_id, dtype=np.int32),
        np.asarray(state_tlbr, dtype=np.float64).reshape(-1, 4),
    )


def synthetic(seed, n_frames=400):
    """Boxes moving across the frame with noise, dropouts, births, deaths,
    crossing paths and low-confidence detections (second association), plus
    slow objects hidden for 20-90 frames, so that lost tracks come back just
    before or just after `lost_track_buffer` expires."""
    rng = np.random.default_rng(seed)
    objects = []
    for i in range(40):
        start = int(rng.integers(1, n_frames - 30))
        size = float(rng.uniform(30, 90))
        slow = i % 3 == 0
        gap_start = start + int(rng.integers(5, 40))
        objects.append(
            {
                "start": start,
                "end": int(
                    min(n_frames, start + rng.integers(15, 160) + (120 if slow else 0))
                ),
                "pos": np.array([rng.uniform(0, 1200), rng.uniform(0, 650)]),
                "vel": rng.uniform(-0.3, 0.3, 2) if slow else rng.uniform(-14, 14, 2),
                "size": np.array([size, size * rng.uniform(0.8, 1.25)]),
                "gap": (gap_start, gap_start + int(rng.integers(20, 91)))
                if slow
                else (0, 0),
            }
        )
    frame, xyxy, conf = [], [], []
    for f in range(1, n_frames + 1):
        for ob in objects:
            if not ob["start"] <= f <= ob["end"] or ob["gap"][0] <= f < ob["gap"][1]:
                continue
            center = ob["pos"] + ob["vel"] * (f - ob["start"])
            if rng.random() < 0.12:  # missed detection: track goes lost
                continue
            center = center + rng.normal(0, 1.5, 2)
            half = ob["size"] / 2 * rng.uniform(0.95, 1.05, 2)
            frame.append(f)
            xyxy.append([*(center - half), *(center + half)])
            # mostly confident; some in (0.1, 0.25) for the second association and
            # some in (0.25, 0.35) that match but never start a track
            conf.append(
                rng.choice(
                    [
                        rng.uniform(0.5, 0.95),
                        rng.uniform(0.11, 0.24),
                        rng.uniform(0.26, 0.34),
                    ],
                    p=[0.8, 0.1, 0.1],
                )
            )
    return (
        np.asarray(frame, dtype=np.int32),
        np.asarray(xyxy, dtype=np.float32),
        np.asarray(conf, dtype=np.float32),
        n_frames,
    )
