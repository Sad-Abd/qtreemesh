"""Tests for QTree internals: Point, neighbor search, balancing, image_preprocess."""

import numpy as np

from qtreemesh import QTree, image_preprocess
from qtreemesh._qtreemesh import Point


def refined_image(size=64):
    """Image with a small bright block that forces local refinement."""
    image = np.full((size, size), 7)
    image[size // 2 - 1 : size // 2 + 3, size // 2 - 1 : size // 2 + 3] = 200
    return image


def test_point_coord_sum():
    point = Point((3.0, 4.0))
    summed = point.coord_sum((1.5, -2.5))
    assert summed.xy_coord == (4.5, 1.5)
    assert summed.x_coord == 4.5
    assert summed.y_coord == 1.5


def test_count_leaves_matches_save_leaves():
    quad = QTree(None, refined_image(), 1)
    assert quad.count_leaves == len(quad.save_leaves())


def test_single_leaf_has_no_neighbors():
    quad = QTree(None, np.full((16, 16), 3), 1)
    leaf = quad.save_leaves()[0]
    assert leaf.north_neighbor() is None
    assert leaf.south_neighbor() is None
    assert leaf.west_neighbor() is None
    assert leaf.east_neighbor() is None


def test_west_neighbor_matches_geometry():
    quad = QTree(None, refined_image(), 1)
    for leaf in quad.save_leaves():
        west = leaf.west_neighbor()
        if leaf.bottom_left_corner.x_coord == 0:
            assert west is None
        else:
            assert west is not None
            assert west.top_right_corner.x_coord == leaf.bottom_left_corner.x_coord
            assert west.bottom_left_corner.y_coord < leaf.top_right_corner.y_coord
            assert west.top_right_corner.y_coord > leaf.bottom_left_corner.y_coord


def test_east_neighbor_matches_geometry():
    quad = QTree(None, refined_image(), 1)
    for leaf in quad.save_leaves():
        east = leaf.east_neighbor()
        if leaf.top_right_corner.x_coord == 64:
            assert east is None
        else:
            assert east is not None
            assert east.bottom_left_corner.x_coord == leaf.top_right_corner.x_coord
            assert east.bottom_left_corner.y_coord < leaf.top_right_corner.y_coord
            assert east.top_right_corner.y_coord > leaf.bottom_left_corner.y_coord


def test_south_neighbor_matches_geometry():
    quad = QTree(None, refined_image(), 1)
    for leaf in quad.save_leaves():
        south = leaf.south_neighbor()
        if leaf.bottom_left_corner.y_coord == 0:
            assert south is None
        else:
            assert south is not None
            assert south.top_right_corner.y_coord == leaf.bottom_left_corner.y_coord
            assert south.bottom_left_corner.x_coord < leaf.top_right_corner.x_coord
            assert south.top_right_corner.x_coord > leaf.bottom_left_corner.x_coord


def test_north_neighbor_matches_geometry():
    quad = QTree(None, refined_image(), 1)
    for leaf in quad.save_leaves():
        north = leaf.north_neighbor()
        if leaf.top_right_corner.y_coord == 64:
            assert north is None
        else:
            assert north is not None
            assert north.bottom_left_corner.y_coord == leaf.top_right_corner.y_coord
            assert north.bottom_left_corner.x_coord < leaf.top_right_corner.x_coord
            assert north.top_right_corner.x_coord > leaf.bottom_left_corner.x_coord


def test_need_split_false_for_single_cell_tree():
    quad = QTree(None, np.full((8, 8), 3), 1)
    leaf = quad.save_leaves()[0]
    assert not QTree.need_split(leaf)
    assert not QTree.need_split(None)


def test_balancing_enforces_two_to_one_ratio():
    quad = QTree(None, refined_image(), 1)
    unbalanced_count = quad.count_leaves
    quad.balancing()
    assert quad.count_leaves >= unbalanced_count
    for leaf in quad.save_leaves():
        assert not QTree.need_split(leaf)
        for neighbor in (
            leaf.north_neighbor(),
            leaf.south_neighbor(),
            leaf.west_neighbor(),
            leaf.east_neighbor(),
        ):
            if neighbor is not None and not neighbor.divided:
                # a leaf neighbor is at most twice the leaf's size
                assert neighbor.dimension * 2 >= leaf.dimension


def test_image_preprocess_leaves_square_power_of_two_unchanged():
    image = np.arange(64).reshape(8, 8)
    assert np.array_equal(image_preprocess(image), image)


def test_image_preprocess_pads_to_square_power_of_two():
    image = np.ones((100, 60))
    out = image_preprocess(image)
    assert out.shape == (128, 128)
    assert np.array_equal(out[:100, :60], image)
    assert out[100:, :].sum() == 0
    assert out[:, 60:].sum() == 0


def test_image_preprocess_pads_rows_then_columns():
    image = np.full((5, 3), 9)
    out = image_preprocess(image)
    assert out.shape == (8, 8)
    assert np.array_equal(out[:5, :3], image)


def test_image_preprocess_pads_portrait_image():
    image = np.full((60, 100), 4)
    out = image_preprocess(image)
    assert out.shape == (128, 128)
    assert np.array_equal(out[:60, :100], image)
    assert out[60:, :].sum() == 0
    assert out[:, 100:].sum() == 0


def test_count_leaves_of_none_is_zero():
    """Documented guard: a None node has no leaves."""
    assert QTree.count_leaves.fget(None) == 0
