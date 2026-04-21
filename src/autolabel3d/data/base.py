"""Abstract base class for data loaders.

Every data source implements this interface. The pipeline is agnostic to
WHERE data comes from — it just calls load_frames() and receives Frame objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from autolabel3d.data.schemas import Frame


class BaseDataLoader(ABC):
    """Abstract data loader that all data sources must implement."""

    @abstractmethod
    def __len__(self) -> int:
        """Total number of frames available."""
        ...

    @abstractmethod
    def load_frames(self) -> Iterator[Frame]:
        """Yield frames one at a time (memory-efficient iterator)."""
        ...

    @abstractmethod
    def get_frame(self, idx: int) -> Frame:
        """Load a specific frame by index."""
        ...
