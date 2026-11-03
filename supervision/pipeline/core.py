from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from typing import Any, Protocol


class PipelineStep(Protocol):
    """
    Protocol defining the interface for pipeline steps.

    All pipeline steps must implement process() method to transform data
    flowing through the pipeline. Optionally, steps can implement filter()
    to skip processing based on conditions.

    Examples:
        ```python
        class CustomStep:
            def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
                # Transform data
                data['processed'] = True
                return data

            def filter(self, data: Dict[str, Any]) -> bool:
                # Optional: filter data
                return data.get('active', True)
        ```
    """

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Process and transform pipeline data.

        Args:
            data: Dictionary containing pipeline data

        Returns:
            Transformed data dictionary
        """
        ...

    def filter(self, data: dict[str, Any]) -> bool:
        """
        Filter data before processing (optional).

        Args:
            data: Dictionary containing pipeline data

        Returns:
            True if data should be processed, False to skip
        """
        ...


class PipelineSource(ABC):
    """
    Abstract base class for pipeline data sources.

    Sources provide data to the pipeline through iteration.
    """

    @abstractmethod
    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        """
        Iterate over source data.

        Yields:
            Dictionary containing source data (e.g., frame, metadata)
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Clean up source resources.
        """
        ...


class PipelineSink(ABC):
    """
    Abstract base class for pipeline data sinks.

    Sinks consume data from the pipeline and perform final operations
    (display, save, callback, etc.).
    """

    @abstractmethod
    def consume(self, data: dict[str, Any]) -> bool | None:
        """
        Consume data from the pipeline.

        Args:
            data: Dictionary containing pipeline data

        Returns:
            False to stop pipeline iteration, True or None to continue
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Clean up sink resources.
        """
        ...


class ThreadBoundary:
    """
    Splits a pipeline in two stages that run in parallel, one frame apart.

    Added with `|` like a step, but it is not one: it does not touch the data.
    Everything before it (the source and the earlier steps) runs in a worker
    thread; everything after it (the later steps and the sink) runs in the
    thread that iterates the pipeline. While the later stage handles frame N,
    the worker already reads and processes frame N + 1. A frame then costs the
    slower of the two stages instead of their sum, which pays off when one
    stage waits on the GPU (inference) and the other is CPU work (tracking).

    ```
    worker:    source -> step 1 -> ... -> step k --+
                                                   v
                                      FIFO queue (queue_size)
                                                   |
    caller:    step k+1 -> ... -> step n -> sink -> yield
    ```

    Guarantees, so that the output is the same as without the boundary:

    - Order: one producer and one consumer on a FIFO queue. Every item comes
      out once and in order; nothing is dropped, reordered or replaced by
      empty results on timeouts.
    - Ownership: each data dict is handed over whole; the worker never
      touches it again once queued. Steps before the boundary must not reuse
      a buffer across frames (sources create a new frame array per read).
    - End of source: the caller gets every item, then the iteration ends.
    - Errors: an exception raised before the boundary (source or early step)
      is raised again in the caller, after the items that preceded it. One
      raised after the boundary propagates as usual and stops the worker.
    - Stopping: leaving the iteration early (break, close, exception) stops
      the worker, which is joined before the source is closed.
    - Memory: the queue is bounded; a slow later stage blocks the worker.

    Only one boundary per pipeline. Steps after it see wall-clock times up to
    `queue_size` frames later than before (e.g. an FPS step measures the
    output rate, as it does without the boundary). Python runs one thread at a
    time (GIL): the overlap comes from code that releases it while it works
    or waits (OpenCV, NumPy, CUDA/TensorRT inference).

    Args:
        queue_size: Frames that can wait between the stages (default 2).
            Larger values absorb jitter at the cost of latency.

    Examples:
        ```python
        import supervision as sv

        pipeline = (
            sv.Pipeline(sv.VideoFileSource("video.mp4"))
            | sv.YOLODetectionStep("yolov8n.pt")   # worker thread
            | sv.ThreadBoundary()
            | sv.ByteTrackerStep()                   # calling thread
            | sv.BoxAnnotatorStep()
        )
        for data in pipeline:
            print(data["detections"].tracker_id)
        ```
    """

    _END = object()

    def __init__(self, queue_size: int = 2):
        if queue_size < 1:
            raise ValueError(f"queue_size must be >= 1, got {queue_size}")
        self.queue_size = queue_size

    def run(
        self, items: Iterable[dict[str, Any]], join_timeout: float = 5.0
    ) -> Generator[dict[str, Any], None, None]:
        """
        Iterate `items` in a worker thread and yield them in this one.

        Args:
            items: The first stage (source plus the steps before the boundary).
            join_timeout: Seconds to wait for the worker when stopping.
        """
        q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        stop = threading.Event()

        def put(item: Any) -> bool:
            # Blocking put that gives up once the caller stopped consuming
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def work() -> None:
            iterator = iter(items)
            try:
                for data in iterator:
                    if not put((None, data)):
                        return
                put((None, self._END))
            except BaseException as error:  # handed over, raised in the caller
                put((error, None))
            finally:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()

        worker = threading.Thread(target=work, name="pipeline-stage-1", daemon=True)
        worker.start()
        try:
            while True:
                error, data = q.get()
                if error is not None:
                    raise error
                if data is self._END:
                    return
                yield data
        finally:
            stop.set()
            while True:  # unblock a worker waiting to put
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            worker.join(join_timeout)


class Pipeline:
    """
    Modern pipeline for composable video processing.

    Pipeline allows chaining multiple processing steps using the `|` operator,
    providing a clean and expressive API for building video processing workflows.

    Attributes:
        source: Data source for the pipeline
        steps: List of processing steps
        sink: Optional data sink

    Examples:
        ```python
        import supervision as sv

        # Create pipeline with source
        pipeline = sv.Pipeline(source=sv.WebcamSource())

        # Add steps using | operator
        pipeline = pipeline | sv.YOLODetectionStep("yolov8n.pt") | sv.BoxAnnotatorStep()

        # Add sink
        pipeline = pipeline | sv.DisplaySink("Detection")

        # Run pipeline
        for data in pipeline:
            # Access processed data
            frame = data['frame']
            detections = data.get('detections')
        ```
    """

    def __init__(self, source: PipelineSource | None = None):
        """
        Initialize pipeline with optional source.

        Args:
            source: Data source for the pipeline
        """
        self.source = source
        self.steps: list[PipelineStep] = []
        self.sink: PipelineSink | None = None
        # Steps before the boundary (steps[:index]) run in its worker thread
        self.thread_boundary: ThreadBoundary | None = None
        self.thread_boundary_index = 0

    def add_step(self, step: PipelineStep | PipelineSink | ThreadBoundary) -> Pipeline:
        """
        Add a processing step, a sink or a `ThreadBoundary` to the pipeline.

        Args:
            step: Pipeline step, sink or thread boundary to add

        Returns:
            Self for method chaining

        Raises:
            ValueError: If a second `ThreadBoundary` is added
        """
        if isinstance(step, PipelineSink):
            self.sink = step
        elif isinstance(step, ThreadBoundary):
            if self.thread_boundary is not None:
                raise ValueError("A pipeline accepts only one ThreadBoundary")
            self.thread_boundary = step
            self.thread_boundary_index = len(self.steps)
        else:
            self.steps.append(step)
        return self

    def __or__(self, step: PipelineStep | PipelineSink | ThreadBoundary) -> Pipeline:
        """
        Add step using | operator (Unix pipe style).

        Args:
            step: Pipeline step or sink to add

        Returns:
            Self for method chaining

        Examples:
            ```python
            pipeline = Pipeline(source) | step1 | step2 | sink
            ```
        """
        return self.add_step(step)

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        """
        Iterate through pipeline, yielding processed data.

        Yields:
            Dictionary containing processed data from each iteration

        Raises:
            ValueError: If no source is configured
        """
        if self.source is None:
            raise ValueError("Pipeline requires a source to iterate")

        boundary = None
        if self.thread_boundary is None:
            stream = _run_steps(self.source, self.steps)
        else:
            k = self.thread_boundary_index
            boundary = self.thread_boundary.run(_run_steps(self.source, self.steps[:k]))
            stream = _run_steps(boundary, self.steps[k:])
        try:
            for data in stream:
                # Send to sink if configured
                if self.sink:
                    result = self.sink.consume(data)
                    # If sink returns False, stop iteration
                    if result is False:
                        break

                # Yield processed data
                yield data

        finally:
            # Stop the boundary's worker before closing the source. Closed
            # explicitly: after an error in a later step the traceback keeps
            # the boundary generator alive, so closing `stream` won't reach it
            stream.close()
            if boundary is not None:
                boundary.close()
            # Cleanup resources
            self.close()

    def run(self) -> None:
        """
        Run the pipeline without yielding data.

        This is useful when using sinks that consume all output
        (e.g., DisplaySink, VideoSink).
        """
        for _ in self:
            pass

    def close(self) -> None:
        """
        Close and cleanup all pipeline resources.
        """
        if self.source:
            self.source.close()
        if self.sink:
            self.sink.close()

    def __enter__(self) -> Pipeline:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def _run_steps(
    items: Iterable[dict[str, Any]], steps: list[PipelineStep]
) -> Generator[dict[str, Any], None, None]:
    """Pass each item through `steps` in order (a step whose `filter` returns
    False is skipped for that item) and yield it."""
    for data in items:
        for step in steps:
            if hasattr(step, "filter") and callable(step.filter):
                if not step.filter(data):
                    continue
            data = step.process(data)
        yield data
