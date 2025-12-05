"""Core workspace for managing geometric objects across coordinate frames."""

from dataclasses import dataclass, field
from typing import Any, Dict

from pytransform3d.transform_manager import TransformManager


@dataclass
class Workspace:
    """Container for geometric objects organized by frames and configurations.

    The workspace manages:
    - Scenes: Collections of geometric objects with coordinates in specific frames
    - Configurations: Spatial arrangements defining how frames relate via transforms

    Attributes:
        scenes: Mapping of {frame_name: {object_id: object_data, ...}}
                Each frame contains objects with their geometric data.
                The same object_id can appear in multiple frames.
        configurations: Mapping of {configuration_name: TransformManager, ...}
                        Each configuration represents a different spatial arrangement
                        of frames connected by rigid transforms.
    """

    scenes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    configurations: Dict[str, TransformManager] = field(default_factory=dict)
