"""Tests for the max_size option of QTree.

max_size caps the side of every leaf cell in pixels: a cell larger than
max_size splits even when its pixels are homogeneous. Without it, a
homogeneous region stays a single cell of the full image size.
"""

import numpy as np
import pytest

from qtreemesh import QTree


def leaves_of(quad):
    return quad.save_leaves()


def test_no_max_size_keeps_homogeneous_region_whole():
    quad = QTree(None, np.full((64, 64), 7), 1)
    leaves = leaves_of(quad)
    assert len(leaves) == 1
    assert leaves[0].dimension == 64


def test_max_size_splits_homogeneous_region():
    quad = QTree(None, np.full((64, 64), 7), 1, max_size=8)
    leaves = leaves_of(quad)
    assert len(leaves) == 64  # (64 / 8) ** 2
    assert all(leaf.dimension == 8 for leaf in leaves)


def test_max_size_is_respected_within_power_of_two():
    """Splits halve the cell, so the side is the largest power of 2 <= max_size."""
    quad = QTree(None, np.full((64, 64), 7), 1, max_size=12)
    leaves = leaves_of(quad)
    assert all(leaf.dimension == 8 for leaf in leaves)


def test_max_size_does_not_block_intensity_refinement():
    """Fine detail still refines to cells far smaller than max_size."""
    image = np.full((64, 64), 7)
    image[30:34, 30:34] = 200
    quad = QTree(None, image, 1, max_size=32)
    leaves = leaves_of(quad)
    dimensions = {leaf.dimension for leaf in leaves}
    assert min(dimensions) < 32
    assert max(dimensions) <= 32


def test_max_size_is_in_pixels_not_real_units():
    """With scale = 2, max_size = 8 caps cells at 8 px = 16 real units."""
    quad = QTree(None, np.full((64, 64), 7), 1, scale=2.0, max_size=8)
    leaves = leaves_of(quad)
    assert all(leaf.dimension == 8 for leaf in leaves)
    span = leaves[0].top_right_corner.x_coord - leaves[0].bottom_left_corner.x_coord
    assert span == 16


def test_max_size_default_is_backward_compatible():
    image = np.zeros((16, 16))
    image[4, 4] = 255
    explicit = QTree(None, image, 1, max_size=None)
    default = QTree(None, image, 1)
    assert len(explicit.save_leaves()) == len(default.save_leaves())


def test_max_size_rejects_nonpositive():
    with pytest.raises(ValueError):
        QTree(None, np.full((8, 8), 7), 1, max_size=0)
