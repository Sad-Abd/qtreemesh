"""Tests for input validation and structural invariants of the mesh."""

import numpy as np
import pytest

from qtreemesh import QTree, QTreeMesh, image_preprocess


def disk_image(size=32):
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size), dtype=int)
    image[((xx - size // 2) ** 2 + (yy - size // 2) ** 2) < size**2 // 16] = 5
    return image


def test_leaf_depth_matches_cell_size():
    mesh = QTreeMesh.from_image(disk_image(), crit=1)
    padded = mesh.quad_tree.array.shape[0]
    root = mesh.quad_tree
    assert root.depth == 0
    for leaf in mesh.leaves:
        expected = int(round(np.log2(padded / leaf.dimension)))
        assert leaf.depth == expected


def test_internal_node_depth_unchanged_by_splitting():
    quad = QTree(None, disk_image(), 1)
    divided = [quad]
    while divided:
        node = divided.pop()
        assert node.depth == int(round(np.log2(quad.array.shape[0] / node.dimension)))
        if node.divided:
            divided += [node.north_west, node.north_east, node.south_west, node.south_east]


def test_unbalanced_mesh_raises_clear_error():
    with pytest.raises(ValueError, match="balanced"):
        QTreeMesh.from_image(disk_image(), crit=1, balancing=False)


def test_image_preprocess_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        image_preprocess(np.zeros((8, 8, 3), dtype=int))


def test_qtree_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        QTree(None, np.zeros((8, 8, 3), dtype=int), 1)


def test_mode_detection_covers_every_pattern():
    for bits in np.ndindex(2, 2, 2, 2):
        mode = [bool(b) for b in bits]
        result = QTreeMesh.mode_detection(mode)
        assert isinstance(result, list) and len(result) == 2
        assert result[0] in (1, 2, 3, 4, 5, 6)
        assert result[1] % 90 == 0
