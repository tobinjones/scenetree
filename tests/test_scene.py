"""Tests for the Scene class."""

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
