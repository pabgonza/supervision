"""
Tests for ThreadBoundary: a pipeline split in two threads must give the same
output as the same pipeline without the boundary.
"""

import random
import threading
import time

import pytest

import supervision as sv

WORKER = "pipeline-stage-1"


def worker_alive() -> bool:
    return any(t.name == WORKER and t.is_alive() for t in threading.enumerate())


class CountingSource(sv.PipelineSource):
    def __init__(self, n: int, fail_at: int | None = None):
        self.n = n
        self.fail_at = fail_at
        self.produced = 0
        self.closed = False
        self.worker_alive_at_close = None

    def __iter__(self):
        for i in range(self.n):
            if i == self.fail_at:
                raise RuntimeError(f"source failed at {i}")
            self.produced += 1
            yield {"i": i, "trace": []}

    def close(self):
        self.closed = True
        self.worker_alive_at_close = worker_alive()


class Step:
    """Appends its name to data['trace'] after a random short delay."""

    def __init__(self, name: str, max_delay: float = 0.002, seed: int = 0):
        self.name = name
        self.max_delay = max_delay
        self.rng = random.Random(seed)

    def process(self, data):
        time.sleep(self.rng.uniform(0, self.max_delay))
        data["trace"].append(self.name)
        return data


class EvenOnly(Step):
    def filter(self, data):
        return data["i"] % 2 == 0


class FailAt:
    def __init__(self, i: int):
        self.i = i

    def process(self, data):
        if data["i"] == self.i:
            raise ValueError(f"step failed at {self.i}")
        return data


class StopAfter(sv.PipelineSink):
    def __init__(self, n: int):
        self.n = n
        self.seen = 0

    def consume(self, data):
        self.seen += 1
        return self.seen < self.n

    def close(self):
        pass


def build(source, boundary_at: int | None, steps):
    pipeline = sv.Pipeline(source)
    for k, step in enumerate(steps):
        if k == boundary_at:
            pipeline = pipeline | sv.ThreadBoundary()
        pipeline = pipeline | step
    if boundary_at == len(steps):
        pipeline = pipeline | sv.ThreadBoundary()
    return pipeline


def make_steps(seed: int):
    return [
        Step("a", seed=seed),
        EvenOnly("even", seed=seed + 1),
        Step("b", seed=seed + 2),
        Step("c", seed=seed + 3),
    ]


@pytest.mark.parametrize("boundary_at", [0, 1, 2, 4])
def test_same_output_in_order(boundary_at):
    n = 300
    expected = [
        (d["i"], tuple(d["trace"]))
        for d in build(CountingSource(n), None, make_steps(0))
    ]
    source = CountingSource(n)
    got = [
        (d["i"], tuple(d["trace"])) for d in build(source, boundary_at, make_steps(0))
    ]
    # Every item once, in order, with the same steps applied (filter included)
    assert got == expected
    assert [i for i, _ in got] == list(range(n))
    assert source.closed and not source.worker_alive_at_close
    assert not worker_alive()


def test_error_before_boundary_after_preceding_items():
    source = CountingSource(100, fail_at=40)
    seen = []
    with pytest.raises(RuntimeError, match="source failed at 40"):
        for data in build(source, 1, [Step("a"), Step("b")]):
            seen.append(data["i"])
    assert seen == list(range(40))
    assert source.closed and not worker_alive()


def test_error_in_early_step_raised_in_caller():
    source = CountingSource(100)
    seen = []
    with pytest.raises(ValueError, match="step failed at 25"):
        for data in build(source, 1, [FailAt(25), Step("b")]):
            seen.append(data["i"])
    assert seen == list(range(25))
    assert source.closed and not worker_alive()


def test_error_after_boundary_stops_worker():
    source = CountingSource(10_000)
    with pytest.raises(ValueError, match="step failed at 10"):
        for _ in build(source, 1, [Step("a", max_delay=0), FailAt(10)]):
            pass
    assert source.closed and not source.worker_alive_at_close
    assert source.produced < 10_000
    assert not worker_alive()


def test_early_break_stops_worker():
    source = CountingSource(10_000)
    for data in build(source, 1, [Step("a", max_delay=0), Step("b")]):
        if data["i"] == 5:
            break
    assert source.closed and not source.worker_alive_at_close
    assert source.produced < 10_000
    assert not worker_alive()


def test_sink_stop_stops_worker():
    source = CountingSource(10_000)
    pipeline = build(source, 1, [Step("a", max_delay=0)]) | StopAfter(7)
    assert len(list(pipeline)) == 6
    assert source.closed and not source.worker_alive_at_close


def test_queue_is_bounded():
    queue_size = 2
    source = CountingSource(1000)

    class Slow:
        def process(self, data):
            time.sleep(0.01)
            # Queued items plus one in the worker's hands, at most
            assert source.produced - (data["i"] + 1) <= queue_size + 1
            return data

    pipeline = sv.Pipeline(source) | sv.ThreadBoundary(queue_size) | Slow()
    for data in pipeline:
        if data["i"] == 30:
            break


def test_end_of_source_yields_last_item():
    got = [d["i"] for d in build(CountingSource(1), 0, [Step("a")])]
    assert got == [0]
    assert [d for d in build(CountingSource(0), 0, [Step("a")])] == []


def test_second_boundary_rejected():
    pipeline = sv.Pipeline(CountingSource(1)) | sv.ThreadBoundary()
    with pytest.raises(ValueError, match="only one ThreadBoundary"):
        pipeline | sv.ThreadBoundary()


def test_queue_size_must_be_positive():
    with pytest.raises(ValueError):
        sv.ThreadBoundary(queue_size=0)
