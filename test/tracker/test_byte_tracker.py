import numpy as np
import pytest

import supervision as sv
from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter


@pytest.mark.parametrize(
    "detections, expected_results",
    [
        (
            [
                sv.Detections(
                    xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
                    class_id=np.array([1, 1]),
                    confidence=np.array([1, 1]),
                ),
                sv.Detections(
                    xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
                    class_id=np.array([1, 1]),
                    confidence=np.array([1, 1]),
                ),
            ],
            sv.Detections(
                xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
                class_id=np.array([1, 1]),
                confidence=np.array([1, 1]),
                tracker_id=np.array([1, 2]),
            ),
        ),
    ],
)
def test_byte_tracker(
    detections: list[sv.Detections],
    expected_results: sv.Detections,
) -> None:
    byte_tracker = sv.ByteTrack()
    tracked_detections = [byte_tracker.update_with_detections(d) for d in detections]
    assert tracked_detections[-1] == expected_results


BOX = 50


def _frame(centers: list[tuple[float, float]]) -> sv.Detections:
    """Detections with square BOX-sized boxes centered on the given points."""
    if not centers:
        return sv.Detections.empty()
    xy = np.array(centers, dtype=float)
    half = BOX / 2
    return sv.Detections(
        xyxy=np.hstack([xy - half, xy + half]),
        confidence=np.full(len(xy), 0.9),
        class_id=np.zeros(len(xy), dtype=int),
    )


def _ids_per_object(
    tracker: sv.ByteTrack, frames: list[dict[str, tuple[float, float]]]
) -> dict[str, list[int]]:
    """Run the tracker and collect, per named object, the id seen on each frame
    (-1 when the object had no tracked box that frame)."""
    seen: dict[str, list[int]] = {}
    for objects in frames:
        names = list(objects)
        tracked = tracker.update_with_detections(_frame([objects[n] for n in names]))
        for name in names:
            center = np.array(objects[name])
            ids = [
                int(tid)
                for box, tid in zip(tracked.xyxy, tracked.tracker_id)
                if np.allclose((box[:2] + box[2:]) / 2, center, atol=1)
            ]
            seen.setdefault(name, []).append(ids[0] if ids else -1)
    return seen


def _warm_up_frames() -> list[dict[str, tuple[float, float]]]:
    """Three lanes whose objects accelerate from 5 to 30 px/frame."""
    frames, x = [], 0.0
    for i in range(70):
        x += min(5 + 1.25 * i, 30)
        frames.append({f"warm{lane}": (x, 100.0 + 100 * lane) for lane in range(3)})
    return frames


def _fast_frames(start: int, lanes: range) -> list[dict[str, tuple[float, float]]]:
    """Objects already moving at 30 px/frame from their first frame."""
    return [
        {f"fast{lane}": (start + 30.0 * i, 100.0 + 100 * lane) for lane in lanes}
        for i in range(15)
    ]


def test_kalman_initiate_velocity() -> None:
    kalman = KalmanFilter()
    measurement = np.array([100.0, 100.0, 1.0, 50.0])
    mean, _ = kalman.initiate(measurement)
    assert np.all(mean[4:] == 0)
    mean, _ = kalman.initiate(measurement, np.array([30.0, 0.0]))
    assert np.allclose(mean[4:], [30, 0, 0, 0])


@pytest.mark.parametrize("alpha", [0, -0.1, 1.5])
def test_velocity_prior_alpha_is_validated(alpha: float) -> None:
    with pytest.raises(ValueError):
        sv.ByteTrack(velocity_prior_alpha=alpha)


def test_default_is_unchanged_without_prior() -> None:
    frames = _warm_up_frames() + _fast_frames(0, range(3, 5))
    default = _ids_per_object(sv.ByteTrack(), frames)
    disabled = _ids_per_object(sv.ByteTrack(velocity_prior_alpha=None), frames)
    assert default == disabled


def test_fast_object_without_prior_is_not_tracked() -> None:
    # born after frame 1, where ByteTrack confirms every track right away
    tracker = sv.ByteTrack()
    _ids_per_object(tracker, _warm_up_frames())
    ids = _ids_per_object(tracker, _fast_frames(0, range(3, 4)))["fast3"][1:]
    assert -1 in ids or len(set(ids)) > 1


def test_velocity_prior_tracks_fast_objects() -> None:
    tracker = sv.ByteTrack(velocity_prior_alpha=0.1)
    _ids_per_object(tracker, _warm_up_frames())
    assert np.allclose(tracker.velocity_prior, [30, 0], atol=3)

    seen = _ids_per_object(tracker, _fast_frames(0, range(3, 5)))
    for name in ("fast3", "fast4"):
        ids = seen[name][1:]  # a new track is only reported from its 2nd frame
        assert -1 not in ids and len(set(ids)) == 1, (name, seen[name])


def test_velocity_prior_is_held_and_reset() -> None:
    tracker = sv.ByteTrack(velocity_prior_alpha=0.1)
    _ids_per_object(tracker, _warm_up_frames())
    prior = tracker.velocity_prior.copy()

    # only a brand-new detection: no confirmed track with history
    tracker.update_with_detections(_frame([(5000.0, 5000.0)]))
    assert np.array_equal(tracker.velocity_prior, prior)

    tracker.reset()
    assert np.array_equal(tracker.velocity_prior, [0, 0])


def test_slow_objects_keep_one_id_with_prior() -> None:
    frames = [
        {f"slow{lane}": (100.0 + 2 * i, 100.0 + 100 * lane) for lane in range(3)}
        for i in range(30)
    ]
    seen = _ids_per_object(sv.ByteTrack(velocity_prior_alpha=0.1), frames)
    ids = {name: set(v[1:]) for name, v in seen.items()}
    assert all(len(v) == 1 and -1 not in v for v in ids.values())
    assert len({next(iter(v)) for v in ids.values()}) == 3


def test_step_forwards_velocity_prior_alpha() -> None:
    step = sv.ByteTrackerStep(velocity_prior_alpha=0.1)
    assert step.tracker.velocity_prior_alpha == 0.1
