"""Tests for gradient-based refinement of the quadtree."""

import numpy as np
import pytest

from qtreemesh import QTree, QTreeMesh


def step_image(size=64):
    """Two uniform halves separated by an intensity jump of 50."""
    image = np.zeros((size, size), dtype=int)
    image[:, size // 2 :] = 150
    return image


def test_grad_crit_refines_across_steep_edges():
    image = step_image()
    coarse = QTreeMesh.from_image(image, crit=200)
    assert len(coarse.elements) == 1

    refined = QTreeMesh.from_image(image, crit=200, grad_crit=20)
    assert len(refined.elements) > 1
    assert all(np.unique(leaf.array).size == 1 for leaf in refined.leaves)


def test_grad_crit_leaves_smooth_regions_coarse():
    size = 64
    ramp = np.tile(np.arange(size, dtype=int) * 4, (size, 1))  # adjacent diff 4
    mesh = QTreeMesh.from_image(ramp, crit=255, grad_crit=5)
    assert len(mesh.elements) == 1


def test_grad_crit_respects_max_size():
    image = step_image()
    mesh = QTreeMesh.from_image(image, crit=200, grad_crit=20, max_size=8)
    assert max(e.element_type[2] for e in mesh.elements) <= 8.0


def test_grad_crit_validation():
    with pytest.raises(ValueError):
        QTree(None, step_image(), 1, grad_crit=0)
    with pytest.raises(ValueError):
        QTree(None, step_image(), 1, grad_crit=-1.0)
    with pytest.raises(ValueError):
        QTree(None, step_image(), 1, grad_crit=np.inf)
    with pytest.raises(ValueError):
        QTree(None, np.zeros((8, 8), dtype=int), 1, label_mode=True, grad_crit=5.0)


def test_grad_crit_through_from_image_matches_manual():
    image = step_image()
    mesh = QTreeMesh.from_image(image, crit=200, grad_crit=20)
    quad = QTree(None, image, 200, grad_crit=20)
    manual = QTreeMesh(quad)
    manual.create_elements()
    assert len(mesh.elements) == len(manual.elements)
    for a, b in zip(mesh.elements, manual.elements):
        assert a.nodes_numbers == b.nodes_numbers
