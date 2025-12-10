"""Tests for the Scene class."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from skspatial.objects import Line, Point, Points

from scenetree import Workspace


class TestSceneDictInterface:
    """Tests for Scene's dict-like interface."""

    def test_setitem_and_getitem(self) -> None:
        """scene[key] = value and scene[key] work correctly."""
        ws = Workspace()
        scene = ws.create_scene("test")
        p = Point([1, 2, 3])
        scene["QP.F1"] = p
        assert scene["QP.F1"] is p

    def test_delitem(self) -> None:
        """Del scene[key] removes the object."""
        ws = Workspace()
        scene = ws.create_scene("test", {"QP.F1": Point([1, 2, 3])})
        del scene["QP.F1"]
        assert "QP.F1" not in scene

    def test_delitem_nonexistent_raises(self) -> None:
        """Deleting nonexistent object raises KeyError."""
        ws = Workspace()
        scene = ws.create_scene("test")
        with pytest.raises(KeyError):
            del scene["nonexistent"]

    def test_contains(self) -> None:
        """'key' in scene works correctly."""
        ws = Workspace()
        scene = ws.create_scene("test", {"QP.F1": Point([1, 2, 3])})
        assert "QP.F1" in scene
        assert "nonexistent" not in scene

    def test_iter(self) -> None:
        """For key in scene yields object IDs."""
        ws = Workspace()
        scene = ws.create_scene("test", {"a": Point([1, 2, 3]), "b": Point([4, 5, 6])})
        assert set(scene) == {"a", "b"}

    def test_len(self) -> None:
        """len(scene) returns object count."""
        ws = Workspace()
        scene = ws.create_scene("test", {"a": Point([1, 2, 3]), "b": Point([4, 5, 6])})
        assert len(scene) == 2

    def test_items(self) -> None:
        """scene.items() returns key-value pairs."""
        ws = Workspace()
        p1 = Point([1, 2, 3])
        p2 = Point([4, 5, 6])
        scene = ws.create_scene("test", {"a": p1, "b": p2})
        items = dict(scene.items())
        assert items == {"a": p1, "b": p2}

    def test_update(self) -> None:
        """scene.update() adds multiple objects."""
        ws = Workspace()
        scene = ws.create_scene("test")
        scene.update({"a": Point([1, 2, 3]), "b": Point([4, 5, 6])})
        assert len(scene) == 2

    def test_ior_operator(self) -> None:
        """Scene |= dict adds multiple objects."""
        ws = Workspace()
        scene = ws.create_scene("test")
        scene |= {"a": Point([1, 2, 3]), "b": Point([4, 5, 6])}
        assert len(scene) == 2


class TestSceneGetPoint:
    """Tests for Scene.get_point()."""

    def test_get_point_from_point(self) -> None:
        """get_point() returns coordinates for Point objects."""
        ws = Workspace()
        scene = ws.create_scene("test", {"p": Point([1.0, 2.0, 3.0])})
        result = scene.get_point("p")
        assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_get_point_from_points(self) -> None:
        """get_point() returns centroid for Points objects."""
        ws = Workspace()
        scene = ws.create_scene("test", {"p": Points([[0, 0, 0], [2, 4, 6]])})
        result = scene.get_point("p")
        assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_get_point_nonexistent_raises(self) -> None:
        """get_point() raises KeyError for nonexistent object."""
        ws = Workspace()
        scene = ws.create_scene("test")
        with pytest.raises(KeyError):
            scene.get_point("nonexistent")

    def test_get_point_wrong_type_raises(self) -> None:
        """get_point() raises TypeError for non-Point/Points objects."""
        ws = Workspace()
        scene = ws.create_scene("test", {"line": Line([0, 0, 0], [1, 0, 0])})
        with pytest.raises(TypeError, match="Expected Point or Points"):
            scene.get_point("line")


class TestSceneGetMeanPoints:
    """Tests for Scene.get_mean_points()."""

    def test_get_mean_points_mixed(self) -> None:
        """get_mean_points() returns dict of positions for Point/Points objects."""
        ws = Workspace()
        scene = ws.create_scene(
            "test",
            {
                "a": Point([1, 2, 3]),
                "b": Points([[0, 0, 0], [2, 4, 6]]),
                "line": Line([0, 0, 0], [1, 0, 0]),  # Should be excluded
            },
        )
        result = scene.get_mean_points()
        assert set(result.keys()) == {"a", "b"}
        assert_array_almost_equal(result["a"], [1, 2, 3])
        assert_array_almost_equal(result["b"], [1, 2, 3])

    def test_get_mean_points_empty_scene(self) -> None:
        """get_mean_points() returns empty dict for empty scene."""
        ws = Workspace()
        scene = ws.create_scene("test")
        assert scene.get_mean_points() == {}


class TestSceneAddPointsFromObservations:
    """Tests for Scene.add_points_from_observations()."""

    def test_single_observation_per_id(self) -> None:
        """Single observations create Points with one point each."""
        ws = Workspace()
        scene = ws.create_scene("test")
        scene.add_points_from_observations(
            [
                ("a", [1, 2, 3]),
                ("b", [4, 5, 6]),
            ],
        )
        assert len(scene) == 2
        assert_array_almost_equal(np.asarray(scene["a"]), [[1, 2, 3]])
        assert_array_almost_equal(np.asarray(scene["b"]), [[4, 5, 6]])

    def test_multiple_observations_coalesced(self) -> None:
        """Multiple observations with same ID are coalesced into single Points."""
        ws = Workspace()
        scene = ws.create_scene("test")
        scene.add_points_from_observations(
            [
                ("a", [1, 2, 3]),
                ("a", [1.1, 2.1, 3.1]),
                ("a", [0.9, 1.9, 2.9]),
            ],
        )
        assert len(scene) == 1
        points = np.asarray(scene["a"])
        assert points.shape == (3, 3)

    def test_empty_observations(self) -> None:
        """Empty observations iterable does nothing."""
        ws = Workspace()
        scene = ws.create_scene("test")
        scene.add_points_from_observations([])
        assert len(scene) == 0

    def test_observations_from_generator(self) -> None:
        """Observations can come from a generator."""
        ws = Workspace()
        scene = ws.create_scene("test")

        def gen():
            yield ("a", [1, 2, 3])
            yield ("b", [4, 5, 6])

        scene.add_points_from_observations(gen())
        assert len(scene) == 2


class TestSceneAddPointsFromCSV:
    """Tests for Scene.add_points_from_csv()."""

    def test_auto_detect_columns_meters(self) -> None:
        """Auto-detect columns with meter units."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ID,x [m],y [m],z [m]\n")
            f.write("F1,-0.000003,0.536642,1.11509\n")
            f.write("F2,0.000001,0.536656,0.6177\n")
            f.write("F3,-0.000023,0.536677,0.2323\n")
            f.write("F4,0.099426,0.536658,0.5447\n")
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            scene.add_points_from_csv(csv_path)

            assert len(scene) == 4
            # Values should be unchanged (already in meters)
            assert_array_almost_equal(
                scene.get_point("F1"),
                [-0.000003, 0.536642, 1.11509],
            )
            assert_array_almost_equal(
                scene.get_point("F2"),
                [0.000001, 0.536656, 0.6177],
            )
        finally:
            Path(csv_path).unlink()

    def test_auto_detect_columns_millimeters(self) -> None:
        """Auto-detect columns with millimeter units and convert to meters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ID,x [mm],y [mm],z [mm]\n")
            f.write("F1,1000,2000,3000\n")
            f.write("F2,500,750,1250\n")
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            scene.add_points_from_csv(csv_path)

            assert len(scene) == 2
            # Values should be converted to meters
            assert_array_almost_equal(scene.get_point("F1"), [1.0, 2.0, 3.0])
            assert_array_almost_equal(scene.get_point("F2"), [0.5, 0.75, 1.25])
        finally:
            Path(csv_path).unlink()

    def test_auto_detect_columns_centimeters(self) -> None:
        """Auto-detect columns with centimeter units and convert to meters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ID,x [cm],y [cm],z [cm]\n")
            f.write("F1,100,200,300\n")
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            scene.add_points_from_csv(csv_path)

            assert_array_almost_equal(scene.get_point("F1"), [1.0, 2.0, 3.0])
        finally:
            Path(csv_path).unlink()

    def test_custom_columns_with_explicit_units(self) -> None:
        """Use custom column names with explicit units parameter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,foo,bar,qux\n")
            f.write("P1,1000,2000,3000\n")
            f.write("P2,500,750,1250\n")
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            scene.add_points_from_csv(
                csv_path,
                id_column="name",
                coord_columns=("foo", "bar", "qux"),
                coord_units="mm",
            )

            assert len(scene) == 2
            assert_array_almost_equal(scene.get_point("P1"), [1.0, 2.0, 3.0])
            assert_array_almost_equal(scene.get_point("P2"), [0.5, 0.75, 1.25])
        finally:
            Path(csv_path).unlink()

    def test_custom_columns_with_units_in_name(self) -> None:
        """Use custom column names where units are embedded in column names."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,x [mm],y [mm],z [mm]\n")
            f.write("P1,1000,2000,3000\n")
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            scene.add_points_from_csv(
                csv_path,
                id_column="name",
                coord_columns=("x [mm]", "y [mm]", "z [mm]"),
            )

            assert_array_almost_equal(scene.get_point("P1"), [1.0, 2.0, 3.0])
        finally:
            Path(csv_path).unlink()

    def test_case_insensitive_headers(self) -> None:
        """Column names are case insensitive."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,X [M],Y [M],Z [M]\n")
            f.write("F1,1,2,3\n")
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            scene.add_points_from_csv(csv_path)

            assert_array_almost_equal(scene.get_point("F1"), [1, 2, 3])
        finally:
            Path(csv_path).unlink()

    def test_different_bracket_styles(self) -> None:
        """Units can be in [], (), {}, or no brackets."""
        test_cases = [
            "ID,x [m],y (m),z {m}\n",
            "ID,x(m),y[m],z{m}\n",
            "ID,x m,y m,z m\n",
        ]

        for header in test_cases:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                f.write(header)
                f.write("F1,1,2,3\n")
                csv_path = f.name

            try:
                ws = Workspace()
                scene = ws.create_scene("test")
                scene.add_points_from_csv(csv_path)
                assert_array_almost_equal(scene.get_point("F1"), [1, 2, 3])
            finally:
                Path(csv_path).unlink()

    def test_multiple_observations_same_id(self) -> None:
        """Multiple CSV rows with same ID create a Points object."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ID,x [m],y [m],z [m]\n")
            f.write("F1,1,2,3\n")
            f.write("F1,1.1,2.1,3.1\n")
            f.write("F1,0.9,1.9,2.9\n")
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            scene.add_points_from_csv(csv_path)

            assert len(scene) == 1
            points = np.asarray(scene["F1"])
            assert points.shape == (3, 3)
            # Centroid should be close to [1, 2, 3]
            assert_array_almost_equal(scene.get_point("F1"), [1.0, 2.0, 3.0])
        finally:
            Path(csv_path).unlink()

    def test_empty_id_rows_skipped(self) -> None:
        """Rows with empty ID fields are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ID,x [m],y [m],z [m]\n")
            f.write("F1,1,2,3\n")
            f.write(",4,5,6\n")
            f.write("F2,7,8,9\n")
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            scene.add_points_from_csv(csv_path)

            assert len(scene) == 2
            assert "F1" in scene
            assert "F2" in scene
        finally:
            Path(csv_path).unlink()

    def test_missing_id_column_raises(self) -> None:
        """Missing ID column raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("x [m],y [m],z [m]\n")
            f.write("1,2,3\n")
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            with pytest.raises(ValueError, match=r"ID column.*not found"):
                scene.add_points_from_csv(csv_path)
        finally:
            Path(csv_path).unlink()

    def test_missing_coord_column_raises(self) -> None:
        """Missing coordinate column raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ID,x [m],y [m]\n")
            f.write("F1,1,2\n")
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            with pytest.raises(ValueError, match="auto-detect coordinate columns"):
                scene.add_points_from_csv(csv_path)
        finally:
            Path(csv_path).unlink()

    def test_custom_column_not_found_raises(self) -> None:
        """Specifying non-existent custom column raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ID,a,b,c\n")
            f.write("F1,1,2,3\n")
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            with pytest.raises(ValueError, match=r"Coordinate column.*not found"):
                scene.add_points_from_csv(
                    csv_path,
                    coord_columns=("x", "y", "z"),
                    coord_units="m",
                )
        finally:
            Path(csv_path).unlink()

    def test_no_units_in_columns_no_param_raises(self) -> None:
        """Custom columns without units and no coord_units param raises."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ID,a,b,c\n")
            f.write("F1,1,2,3\n")
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            with pytest.raises(ValueError, match="Could not determine units"):
                scene.add_points_from_csv(csv_path, coord_columns=("a", "b", "c"))
        finally:
            Path(csv_path).unlink()

    def test_invalid_units_raises(self) -> None:
        """Invalid units raise ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ID,a,b,c\n")
            f.write("F1,1,2,3\n")
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            with pytest.raises(ValueError, match="Invalid units"):
                scene.add_points_from_csv(
                    csv_path,
                    coord_columns=("a", "b", "c"),
                    coord_units="feet",
                )
        finally:
            Path(csv_path).unlink()

    def test_invalid_coordinate_data_raises(self) -> None:
        """Non-numeric coordinate data raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ID,x [m],y [m],z [m]\n")
            f.write("F1,1,not_a_number,3\n")
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            with pytest.raises(ValueError, match="Invalid coordinate data"):
                scene.add_points_from_csv(csv_path)
        finally:
            Path(csv_path).unlink()

    def test_inconsistent_units_raises(self) -> None:
        """Coordinate columns with different units raise ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ID,x [m],y [mm],z [cm]\n")
            f.write("F1,1,2,3\n")
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            with pytest.raises(ValueError, match="inconsistent units"):
                scene.add_points_from_csv(csv_path)
        finally:
            Path(csv_path).unlink()

    def test_no_header_raises(self) -> None:
        """CSV with no header row raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Empty file
            csv_path = f.name

        try:
            ws = Workspace()
            scene = ws.create_scene("test")
            with pytest.raises(ValueError, match="no header row"):
                scene.add_points_from_csv(csv_path)
        finally:
            Path(csv_path).unlink()


class TestSceneProxyBehavior:
    """Tests for Scene proxy semantics."""

    def test_multiple_proxies_share_data(self) -> None:
        """Multiple Scene proxies for same scene share underlying data."""
        ws = Workspace()
        ws.create_scene("test")
        scene1 = ws["test"]
        scene2 = ws["test"]

        scene1["QP.F1"] = Point([1, 2, 3])
        assert "QP.F1" in scene2

    def test_proxy_reflects_workspace_changes(self) -> None:
        """Scene proxy reflects changes made through workspace."""
        ws = Workspace()
        scene = ws.create_scene("test")
        ws["test"]["QP.F1"] = Point([1, 2, 3])
        assert "QP.F1" in scene
