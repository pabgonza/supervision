from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from supervision.utils import capture as capture_module
from supervision.utils.capture import StreamCapture


class FakeVideoCapture:
    """Delivers numbered frames; after `limit` frames, or when `gate` is set and not
    released, `read()` blocks until `stop` is set."""

    def __init__(self, limit: int, gate: threading.Semaphore | None = None):
        self.limit = limit
        self.gate = gate
        self.served = 0
        self.stop = threading.Event()

    def __call__(self, *args, **kwargs) -> FakeVideoCapture:
        return self

    def isOpened(self) -> bool:
        return True

    def set(self, *args) -> bool:
        return True

    def get(self, *args) -> float:
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        while not self.stop.is_set():
            if self.served < self.limit and (
                self.gate is None or self.gate.acquire(timeout=0.01)
            ):
                self.served += 1
                return True, np.full((2, 2, 3), self.served, dtype=np.uint8)
            if self.gate is None:
                self.stop.wait(0.01)
        return False, None

    def release(self) -> None:
        pass


@pytest.fixture
def make_capture(monkeypatch):
    created: list[tuple[StreamCapture, FakeVideoCapture]] = []

    def make(limit: int, queue_size: int, gate=None) -> StreamCapture:
        fake = FakeVideoCapture(limit, gate)
        monkeypatch.setattr(capture_module.cv2, "VideoCapture", fake)
        cap = StreamCapture("fake://stream", queue_size=queue_size)
        created.append((cap, fake))
        return cap

    yield make
    for cap, fake in created:
        cap.stopped = True
        fake.stop.set()
        if cap.thread.is_alive():
            cap.thread.join(timeout=5)


def wait_until(predicate, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_dropped_frames_counts_discards_without_consumer(make_capture):
    cap = make_capture(limit=10, queue_size=4)
    cap.thread.start()

    assert wait_until(lambda: cap.get_health()["dropped_frames"] == 6)
    time.sleep(0.1)  # no further discards while the stream waits
    assert cap.get_health()["dropped_frames"] == 6
    assert [int(cap.read()[0, 0, 0]) for _ in range(4)] == [7, 8, 9, 10]


def test_dropped_frames_stays_zero_when_consumer_keeps_up(make_capture):
    gate = threading.Semaphore(0)
    cap = make_capture(limit=20, queue_size=4, gate=gate)
    cap.thread.start()

    for expected in range(1, 21):
        gate.release()
        assert int(cap.read()[0, 0, 0]) == expected
    assert cap.get_health()["dropped_frames"] == 0


def test_new_capture_reports_zero_dropped_frames(make_capture):
    health = make_capture(limit=0, queue_size=4).get_health()

    assert health["dropped_frames"] == 0
    assert {
        "is_running",
        "is_capture_open",
        "last_frame_time",
        "reconnect_attempts",
        "total_reconnect_attempts",
        "queue_size",
    } <= health.keys()
