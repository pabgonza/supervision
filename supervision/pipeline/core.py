from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
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

    def add_step(self, step: PipelineStep | PipelineSink) -> Pipeline:
        """
        Add a processing step or sink to the pipeline.

        Args:
            step: Pipeline step or sink to add

        Returns:
            Self for method chaining
        """
        if isinstance(step, PipelineSink):
            self.sink = step
        else:
            self.steps.append(step)
        return self

    def __or__(self, step: PipelineStep | PipelineSink) -> Pipeline:
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

        try:
            for data in self.source:
                # Process through all steps
                for step in self.steps:
                    # Check filter if method exists
                    if hasattr(step, "filter") and callable(step.filter):
                        if not step.filter(data):
                            continue

                    # Process data
                    data = step.process(data)

                # Send to sink if configured
                if self.sink:
                    result = self.sink.consume(data)
                    # If sink returns False, stop iteration
                    if result is False:
                        break

                # Yield processed data
                yield data

        finally:
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
