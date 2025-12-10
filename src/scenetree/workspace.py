"""Core workspace for managing geometric objects across coordinate frames."""

import csv
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from copy import deepcopy
from fnmatch import fnmatch
from pathlib import Path
from types import NotImplementedType
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt
from pytransform3d.transform_manager import TransformManager
from scipy.spatial.transform import Rotation
from skspatial.objects import Line, Point, Points

if TYPE_CHECKING:
    from collections.abc import ItemsView

# Type alias for supported geometric object types
SupportedObject = Point | Points | Line


class Scene:
    """Proxy object providing dict-like access to objects in a scene.

    Scene objects are lightweight proxies that reference data stored in the
    parent Workspace. They should not be stored long-term; retrieve a fresh
    proxy from the workspace when needed.
    """

    def __init__(self, workspace: "Workspace", name: str) -> None:
        """Initialize a scene proxy.

        Args:
            workspace: The parent workspace containing the scene data.
            name: The name of this scene.

        """
        self._workspace = workspace
        self._name = name

    @property
    def name(self) -> str:
        """The scene name."""
        return self._name

    def _get_data(self) -> dict[str, SupportedObject]:
        """Get the underlying data dict, raising KeyError if scene doesn't exist."""
        return self._workspace._scenes[self._name]  # type: ignore[return-value]

    def __getitem__(self, object_id: str) -> SupportedObject:
        """Get an object by ID: scene['QP.F1']"""
        return self._get_data()[object_id]

    def __setitem__(self, object_id: str, data: SupportedObject) -> None:
        """Set an object by ID: scene['QP.F1'] = point"""
        self._get_data()[object_id] = data

    def __delitem__(self, object_id: str) -> None:
        """Delete an object by ID: del scene['QP.F1']"""
        del self._get_data()[object_id]

    def __contains__(self, object_id: object) -> bool:
        """Check if object exists: 'QP.F1' in scene"""
        try:
            return object_id in self._get_data()
        except KeyError:
            return False

    def __iter__(self) -> Iterator[str]:
        """Iterate over object IDs: for obj_id in scene"""
        return iter(self._get_data())

    def __len__(self) -> int:
        """Return number of objects in scene: len(scene)"""
        return len(self._get_data())

    def items(self) -> "ItemsView[str, SupportedObject]":
        """Return dict-like `items`: for obj_id, data in scene.items()"""
        return self._get_data().items()

    def update(self, objects: dict[str, SupportedObject]) -> None:
        """Batch add objects: scene.update({'QP.F1': p1, 'QP.F2': p2})"""
        self._get_data().update(objects)

    def __ior__(self, objects: dict[str, SupportedObject]) -> Self:
        """Batch add objects: scene |= {'QP.F1': p1, 'QP.F2': p2}"""
        self.update(objects)
        return self

    def add_points_from_observations(
        self,
        observations: Iterable[tuple[str, npt.ArrayLike]],
    ) -> None:
        """Add Points objects from an iterable of named point observations.

        Multiple observations with the same object_id are coalesced into a single
        Points object containing all observed coordinates.

        Args:
            observations: An iterable of (object_id, coordinates) tuples, where
                coordinates is an array-like of shape (3,) representing [x, y, z].

        Example:
            scene.points_from_observations([
                ("QP.F1", [1, 2, 3]),
                ("QP.F1", [1.1, 2.1, 3.1]),  # second observation of same point
                ("QP.F2", [4, 5, 6]),
            ])
            # scene["QP.F1"] is now Points([[1, 2, 3], [1.1, 2.1, 3.1]])
            # scene["QP.F2"] is now Points([[4, 5, 6]])

        """
        grouped: dict[str, list[npt.ArrayLike]] = defaultdict(list)
        for object_id, coords in observations:
            grouped[object_id].append(coords)

        data = self._get_data()
        for object_id, coords_list in grouped.items():
            data[object_id] = Points(coords_list)

    def add_points_from_csv(
        self,
        csv_path: str | Path,
        *,
        id_fstring: str = "{ID}",
        coord_columns: tuple[str, str, str] | None = None,
        coord_units: str | None = None,
    ) -> None:
        """Load point observations from a CSV file.

        By default, looks for columns named "ID" for object IDs and coordinate
        columns with format "x [unit]", "y [unit]", "z [unit]" where unit is
        one of mm, cm, or m (case insensitive). Units can be in brackets [],
        parentheses (), curly braces {}, or no brackets.

        Coordinates are converted to meters for internal storage.

        Args:
            csv_path: Path to the CSV file.
            id_fstring: Format string for generating object IDs from row data.
                Defaults to "{ID}". Column names are case-sensitive in the format
                string. Examples: "{Assembly_name} Fiducial {id}", "{ID}".
                The format string is evaluated as id_fstring.format(**row_dict).
            coord_columns: Optional tuple of (x_col, y_col, z_col) column names.
                If provided, these exact names are used instead of auto-detection.
            coord_units: Required if coord_columns is provided and the column
                names don't include units. One of "mm", "cm", or "m" (case
                insensitive). If not provided, units are extracted from column
                headers.

        Raises:
            ValueError: If columns are missing, units can't be determined, or
                units are invalid.
            FileNotFoundError: If the CSV file doesn't exist.
            KeyError: If a column referenced in id_fstring is missing from CSV.

        Example:
            Given a CSV file with columns "ID,x [m],y [m],z [m]":
            scene.add_points_from_csv("points.csv")

            Given a CSV with custom ID format:
            scene.add_points_from_csv(
                "points.csv",
                id_fstring="{Assembly_name} Fiducial {id}"
            )

            Given a CSV with custom column names "name,foo,bar,qux":
            scene.add_points_from_csv(
                "points.csv",
                id_fstring="{name}",
                coord_columns=("foo", "bar", "qux"),
                coord_units="mm"
            )

        """
        csv_path = Path(csv_path)

        # Read CSV
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                msg = "CSV file has no header row"
                raise ValueError(msg)

            # Normalize fieldnames for case-insensitive lookup (for coord columns only)
            fieldnames_lower = {name.lower(): name for name in reader.fieldnames}

            # Resolve coordinate columns and units
            actual_coord_cols, units = self._resolve_coord_columns_and_units(
                fieldnames_lower,
                coord_columns,
                coord_units,
            )

            # Determine conversion factor to meters
            unit_to_meters = {"mm": 0.001, "cm": 0.01, "m": 1.0}
            scale = unit_to_meters[units]

            # Parse rows and build observations
            observations: list[tuple[str, npt.ArrayLike]] = []
            for row in reader:
                # Generate object ID using format string
                try:
                    object_id = id_fstring.format(**row).strip()
                except KeyError as e:
                    msg = f"Column referenced in id_fstring not found in CSV: {e}"
                    raise KeyError(msg) from e

                if not object_id:
                    continue

                try:
                    coords = [
                        float(row[actual_coord_cols[0]]) * scale,
                        float(row[actual_coord_cols[1]]) * scale,
                        float(row[actual_coord_cols[2]]) * scale,
                    ]
                    observations.append((object_id, coords))
                except (ValueError, KeyError) as e:
                    msg = f"Invalid coordinate data for object '{object_id}': {e}"
                    raise ValueError(msg) from e

        # Use existing method to add points
        self.add_points_from_observations(observations)

    def _resolve_coord_columns_and_units(
        self,
        fieldnames_lower: dict[str, str],
        coord_columns: tuple[str, str, str] | None,
        coord_units: str | None,
    ) -> tuple[tuple[str, str, str], str]:
        """Resolve coordinate column names and units for CSV parsing.

        Args:
            fieldnames_lower: Dict mapping lowercase fieldname to actual fieldname.
            coord_columns: User-specified coordinate columns or None.
            coord_units: User-specified units or None.

        Returns:
            A tuple of ((x_col, y_col, z_col), units).

        Raises:
            ValueError: If columns are missing or units can't be determined.

        """
        if coord_columns is not None:
            # User specified exact column names
            # Check all columns exist
            for col in coord_columns:
                if col.lower() not in fieldnames_lower:
                    msg = f"Coordinate column '{col}' not found in CSV"
                    raise ValueError(msg)

            actual_coord_cols = (
                fieldnames_lower[coord_columns[0].lower()],
                fieldnames_lower[coord_columns[1].lower()],
                fieldnames_lower[coord_columns[2].lower()],
            )

            # Try to extract units from column names, or use coord_units
            units = self._extract_units_from_columns(actual_coord_cols)
            if units is None:
                if coord_units is None:
                    msg = (
                        "Could not determine units from column names. "
                        "Please provide coord_units parameter."
                    )
                    raise ValueError(msg)
                units = coord_units.lower()
        else:
            # Auto-detect coordinate columns
            actual_coord_cols, units = self._auto_detect_coord_columns(fieldnames_lower)

        # Validate units
        if units not in {"mm", "cm", "m"}:
            raise ValueError(f"Invalid units '{units}'. Must be one of: mm, cm, m")

        return actual_coord_cols, units

    def _extract_units_from_columns(self, columns: tuple[str, str, str]) -> str | None:
        """Try to extract units from coordinate column names.

        Looks for patterns like "x [m]", "y (cm)", "z {mm}", etc.

        Returns:
            The unit string (mm, cm, or m) if found, None otherwise.

        """
        # Pattern matches: coord [unit], coord (unit), coord {unit}, or coord unit
        # where coord is x/y/z and unit is mm/cm/m
        pattern = r"[xyz]\s*[\[\(\{]?\s*(mm|cm|m)\s*[\]\)\}]?$"

        for col in columns:
            match = re.search(pattern, col.lower())
            if match:
                return match.group(1)
        return None

    def _auto_detect_coord_columns(
        self,
        fieldnames_lower: dict[str, str],
    ) -> tuple[tuple[str, str, str], str]:
        """Auto-detect coordinate columns and units from CSV headers.

        Args:
            fieldnames_lower: Dict mapping lowercase fieldname to actual fieldname.

        Returns:
            A tuple of ((x_col, y_col, z_col), units).

        Raises:
            ValueError: If coordinate columns can't be detected.

        """
        # Pattern to match coordinate columns with units
        # Matches: "x [m]", "y (cm)", "z {mm}", "x [unit]", etc.
        coord_pattern = r"^([xyz])\s*[\[\(\{]?\s*(mm|cm|m)\s*[\]\)\}]?$"

        detected: dict[str, tuple[str, str]] = {}  # coord -> (actual_col_name, unit)

        for lower_name, actual_name in fieldnames_lower.items():
            match = re.match(coord_pattern, lower_name)
            if match:
                coord = match.group(1)  # x, y, or z
                unit = match.group(2)  # mm, cm, or m
                detected[coord] = (actual_name, unit)

        # Check we found all three coordinates
        if set(detected.keys()) != {"x", "y", "z"}:
            msg = (
                "Could not auto-detect coordinate columns. "
                f"Expected x, y, z columns with units (mm/cm/m), found: {list(detected.keys())}"
            )
            raise ValueError(msg)

        # Check all have the same units
        units = {unit for _, unit in detected.values()}
        if len(units) != 1:
            raise ValueError(f"Coordinate columns have inconsistent units: {units}")

        unit = units.pop()
        coord_cols = (detected["x"][0], detected["y"][0], detected["z"][0])

        return coord_cols, unit

    def get_point(self, object_id: str) -> npt.NDArray[np.floating[Any]]:
        """Get a single 3D position for an object.

        For Point objects: returns the point coordinates.
        For Points objects: returns the centroid of all points.

        Args:
            object_id: The object ID to look up.

        Returns:
            A numpy array of shape (3,) with the point coordinates.

        Raises:
            KeyError: If the object doesn't exist.
            TypeError: If the object is not a Point or Points.

        """
        obj = self._get_data()[object_id]
        if isinstance(obj, Point):
            return np.asarray(obj)
        if isinstance(obj, Points):
            return np.asarray(obj.centroid())
        raise TypeError(f"Expected Point or Points, got {type(obj).__name__}")

    def get_mean_points(self) -> dict[str, npt.NDArray[np.floating[Any]]]:
        """Get mean positions for all Point/Points objects in the scene.

        Returns:
            A dict mapping object_id to a numpy array of shape (3,).
            Only includes objects that are Point or Points instances.

        """
        result: dict[str, npt.NDArray[np.floating[Any]]] = {}
        for object_id, obj in self._get_data().items():
            if isinstance(obj, Point):
                result[object_id] = np.asarray(obj)
            elif isinstance(obj, Points):
                result[object_id] = np.asarray(obj.centroid())
        return result


class Configuration:
    """Proxy object for managing transforms between scenes.

    Configuration objects are lightweight proxies that reference a TransformManager
    stored in the parent Workspace. They should not be stored long-term; retrieve
    a fresh proxy from the workspace when needed.
    """

    def __init__(self, workspace: "Workspace", name: str) -> None:
        """Initialize a configuration proxy.

        Args:
            workspace: The parent workspace containing the configuration data.
            name: The name of this configuration.

        """
        self._workspace = workspace
        self._name = name

    @property
    def name(self) -> str:
        """The configuration name."""
        return self._name

    def _get_tm(self) -> TransformManager:
        """Get the underlying TransformManager."""
        return self._workspace._configurations[self._name]

    def connect_by_transform(
        self,
        from_scene: str,
        to_scene: str,
        transform: npt.NDArray[np.floating[Any]],
    ) -> None:
        """Add a transform connecting two scenes.

        Args:
            from_scene: The source scene name.
            to_scene: The destination scene name.
            transform: A 4x4 homogeneous transformation matrix.

        Raises:
            KeyError: If either scene doesn't exist in the workspace.

        """
        if from_scene not in self._workspace._scenes:
            raise KeyError(f"Scene '{from_scene}' does not exist")
        if to_scene not in self._workspace._scenes:
            raise KeyError(f"Scene '{to_scene}' does not exist")
        self._get_tm().add_transform(from_scene, to_scene, transform)

    def get_transform(self, from_scene: str, to_scene: str) -> npt.NDArray[np.floating[Any]]:
        """Get the transform between two scenes.

        If the scenes are not directly connected, the transform will be computed
        by following the path through the transform graph.

        Args:
            from_scene: The source scene name.
            to_scene: The destination scene name.

        Returns:
            A 4x4 homogeneous transformation matrix.

        Raises:
            KeyError: If no path exists between the scenes.

        """
        return self._get_tm().get_transform(from_scene, to_scene)

    def as_transform_manager(self) -> TransformManager:
        """Return a copy of the underlying TransformManager.

        Returns:
            A deep copy of the TransformManager, safe to modify without
            affecting the workspace.

        """
        return deepcopy(self._get_tm())

    def get_graph_png(self) -> bytes:
        """Render the transform graph as a PNG image.

        Requires the 'graph' optional dependency: pip install scenetree[graph]

        Returns:
            PNG image data as bytes.

        Raises:
            ImportError: If pydot is not installed.

        """
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            self._get_tm().write_png(f.name)
            return f.read()

    def connect_by_best_fit_points(
        self,
        from_scene: str,
        to_scene: str,
        object_ids: Iterable[str] | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """Compute and add a best-fit rigid transform between two scenes.

        Finds Point/Points objects shared between the scenes and computes
        the optimal rigid transform (rotation + translation) that aligns
        the from_scene points to the to_scene points using least-squares.

        The computed transform is added to this configuration.

        Args:
            from_scene: The source scene name.
            to_scene: The destination scene name.
            object_ids: Optional subset of object IDs to use for fitting.
                If None, uses all shared Point/Points objects.

        Returns:
            The 4x4 homogeneous transformation matrix.

        Raises:
            KeyError: If either scene doesn't exist.
            ValueError: If fewer than 3 shared points are found.

        """
        from_points_dict = self._workspace[from_scene].get_mean_points()
        to_points_dict = self._workspace[to_scene].get_mean_points()

        # Find shared object IDs
        if object_ids is not None:
            shared_ids = set(object_ids) & from_points_dict.keys() & to_points_dict.keys()
        else:
            shared_ids = from_points_dict.keys() & to_points_dict.keys()

        if len(shared_ids) < 3:
            raise ValueError(f"Need at least 3 shared points for best fit, found {len(shared_ids)}")

        # Build point arrays in consistent order
        shared_ids_list = list(shared_ids)
        from_points = np.array([from_points_dict[k] for k in shared_ids_list])
        to_points = np.array([to_points_dict[k] for k in shared_ids_list])

        # Compute centroids
        from_centroid = from_points.mean(axis=0)
        to_centroid = to_points.mean(axis=0)

        # Center the point clouds
        from_centered = from_points - from_centroid
        to_centered = to_points - to_centroid

        # Find optimal rotation using Kabsch algorithm
        rotation, _ = Rotation.align_vectors(to_centered, from_centered)

        # Build 4x4 homogeneous transform:
        # translate_to @ rotate @ translate_from_inv
        # Which transforms a point p as: T @ p = to_centroid + R @ (p - from_centroid)
        transform = np.eye(4)
        transform[:3, :3] = rotation.as_matrix()
        transform[:3, 3] = to_centroid - rotation.apply(from_centroid)

        # Add to configuration
        self._get_tm().add_transform(from_scene, to_scene, transform)

        return transform

    def view_from(self, reference_scene: str) -> "View":
        """Create a View anchored to a reference scene.

        The View allows accessing objects from other scenes transformed
        into the reference scene's coordinate frame.

        Args:
            reference_scene: The scene whose coordinate frame to use.

        Returns:
            A View object for accessing transformed objects.

        Raises:
            KeyError: If the reference scene doesn't exist.

        """
        if reference_scene not in self._workspace._scenes:
            raise KeyError(f"Scene '{reference_scene}' does not exist")
        return View(self, reference_scene)


class View:
    """A view into the workspace from a specific scene's coordinate frame.

    Allows accessing objects from other scenes, automatically transformed
    into the reference scene's coordinate frame using the configuration's
    transform graph.
    """

    def __init__(self, configuration: Configuration, reference_scene: str) -> None:
        """Initialize a view.

        Args:
            configuration: The configuration providing transforms.
            reference_scene: The scene whose coordinate frame to use.

        """
        self._configuration = configuration
        self._reference_scene = reference_scene

    @property
    def reference_scene(self) -> str:
        """The reference scene name."""
        return self._reference_scene

    def _transform_point(
        self,
        point: npt.NDArray[np.floating[Any]],
        transform: npt.NDArray[np.floating[Any]],
    ) -> npt.NDArray[np.floating[Any]]:
        """Apply a 4x4 homogeneous transform to a 3D point."""
        homogeneous = np.append(point, 1.0)
        transformed = transform @ homogeneous
        return transformed[:3]

    def _transform_points(
        self,
        points: npt.NDArray[np.floating[Any]],
        transform: npt.NDArray[np.floating[Any]],
    ) -> npt.NDArray[np.floating[Any]]:
        """Apply a 4x4 homogeneous transform to an array of 3D points."""
        # points is (n, 3), we need to add homogeneous coordinate
        n = points.shape[0]
        homogeneous = np.hstack([points, np.ones((n, 1))])
        transformed = (transform @ homogeneous.T).T
        return transformed[:, :3]

    def get_object(self, from_scene: str, object_id: str) -> SupportedObject | NotImplementedType:
        """Get an object from another scene, transformed into the reference frame.

        Args:
            from_scene: The scene containing the object.
            object_id: The object ID to retrieve.

        Returns:
            The transformed object (Point or Points), or NotImplemented if
            the object type is not supported.

        Raises:
            KeyError: If the scene or object doesn't exist, or if no transform
                path exists between the scenes.

        """
        # Get the transform from source scene to reference scene
        transform = self._configuration.get_transform(from_scene, self._reference_scene)

        # Get the source object
        source_scene = self._configuration._workspace[from_scene]
        obj = source_scene[object_id]

        if isinstance(obj, Point):
            coords = np.asarray(obj)
            transformed_coords = self._transform_point(coords, transform)
            return Point(transformed_coords)
        if isinstance(obj, Points):
            coords = np.asarray(obj)
            transformed_coords = self._transform_points(coords, transform)
            return Points(transformed_coords)
        return NotImplemented

    def _get_connected_scenes(self) -> list[str]:
        """Get all scenes connected to the reference scene in the transform graph."""
        connected = []
        workspace = self._configuration._workspace
        for scene_name in workspace._scenes:
            if scene_name == self._reference_scene:
                continue
            try:
                self._configuration.get_transform(scene_name, self._reference_scene)
                connected.append(scene_name)
            except KeyError:
                # No path to this scene
                pass
        return connected

    def query(
        self,
        object_query: str = "*",
        from_scenes: Iterable[str] | None = None,
    ) -> dict[str, Points | list[SupportedObject]]:
        """Query objects from multiple scenes, transformed into the reference frame.

        Args:
            object_query: A wildcard pattern to match object IDs (e.g., "QP.*", "*").
                Uses fnmatch-style matching.
            from_scenes: Optional list of scene names to query from. If None,
                queries from all scenes connected to the reference scene.

        Returns:
            A dict mapping object_id to transformed objects. For Point/Points objects,
            all matching points are consolidated into a single Points object. For other
            types, returns a list of transformed objects. Unsupported types are skipped.

        """
        scenes_to_query = self._get_connected_scenes() if from_scenes is None else list(from_scenes)

        # Collect point coordinates separately for consolidation
        point_coords: dict[str, list[npt.NDArray[np.floating[Any]]]] = defaultdict(list)
        other_objects: dict[str, list[SupportedObject]] = defaultdict(list)

        for scene_name in scenes_to_query:
            scene = self._configuration._workspace[scene_name]
            for object_id in scene:
                if fnmatch(object_id, object_query):
                    transformed = self.get_object(scene_name, object_id)
                    if transformed is NotImplemented:
                        continue
                    if isinstance(transformed, Point):
                        point_coords[object_id].append(np.asarray(transformed))
                    elif isinstance(transformed, Points):
                        # Add each point from the Points object
                        point_coords[object_id].extend(np.asarray(transformed))
                    else:
                        other_objects[object_id].append(transformed)

        # Build result: consolidate points into Points objects
        points_objects = {object_id: Points(coords) for object_id, coords in point_coords.items()}

        return points_objects | other_objects


class Workspace:
    """Container for geometric objects organized by scenes and configurations.

    The workspace manages:
    - Scenes: Collections of geometric objects with coordinates in specific frames
    - Configurations: Spatial arrangements defining how frames relate via transforms
    """

    def __init__(self) -> None:
        """Initialize an empty workspace."""
        self._scenes: dict[str, dict[str, Any]] = {}
        self._configurations: dict[str, TransformManager] = {}

    def create_scene(self, name: str, objects: dict[str, SupportedObject] | None = None) -> Scene:
        """Create a new scene and return a proxy to it.

        Args:
            name: The name for the new scene.
            objects: Optional dict of objects to populate the scene with.

        Returns:
            A Scene proxy for the newly created scene.

        Raises:
            ValueError: If a scene with this name already exists.

        """
        if name in self._scenes:
            raise ValueError(f"Scene '{name}' already exists")
        self._scenes[name] = objects.copy() if objects else {}
        return Scene(self, name)

    def create_configuration(self, name: str) -> Configuration:
        """Create a new configuration and return a proxy to it.

        Args:
            name: The name for the new configuration.

        Returns:
            A Configuration proxy for the newly created configuration.

        Raises:
            ValueError: If a configuration with this name already exists.

        """
        if name in self._configurations:
            raise ValueError(f"Configuration '{name}' already exists")
        self._configurations[name] = TransformManager()
        return Configuration(self, name)

    def configuration(self, name: str) -> Configuration:
        """Get a configuration proxy by name.

        Args:
            name: The name of the configuration.

        Returns:
            A Configuration proxy.

        Raises:
            KeyError: If the configuration doesn't exist.

        """
        if name not in self._configurations:
            raise KeyError(
                f"Configuration '{name}' does not exist. Use create_configuration() first.",
            )
        return Configuration(self, name)

    def __getitem__(self, scene: str) -> Scene:
        """Get a scene proxy by name: ws['scene_A']

        Raises:
            KeyError: If the scene doesn't exist (use create_scene first).

        """
        if scene not in self._scenes:
            raise KeyError(f"Scene '{scene}' does not exist. Use create_scene() first.")
        return Scene(self, scene)

    def __contains__(self, scene: object) -> bool:
        """Check if scene exists: 'scene_A' in ws"""
        return scene in self._scenes

    def __iter__(self) -> Iterator[str]:
        """Iterate over scene names: for scene in ws"""
        return iter(self._scenes)
