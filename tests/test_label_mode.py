"""Tests for meshing label maps."""

import numpy as np
import pytest

from qtreemesh import QTree, QTreeMesh, image_preprocess


def label_image(size=32):
    image = np.zeros((size, size), dtype=int)
    image[size // 4 : size // 2, size // 4 : 3 * size // 4] = 1
    image[size // 2 : 3 * size // 4, size // 4 : 3 * size // 4] = 2
    return image


def test_label_mode_splits_adjacent_labels():
    image = label_image()
    intensity_mesh = QTreeMesh.from_image(image, crit=1)
    # with crit = 1 the labels 0 and 1 are numerically "close enough" and the
    # intensity path keeps cells that mix them
    mixed = [
        leaf for leaf in intensity_mesh.leaves if np.unique(leaf.array).size > 1
    ]
    assert mixed

    mesh = QTreeMesh.from_image(image, label_mode=True)
    labels = mesh.element_labels(strict=True)
    assert {0, 1, 2} <= set(labels.tolist())
    assert all(np.unique(leaf.array).size == 1 for leaf in mesh.leaves)


def test_label_mode_matches_homogeneity_expectation():
    image = label_image()
    quad = QTree(None, image_preprocess(image), 1, label_mode=True)
    mesh = QTreeMesh(quad)
    mesh.create_elements()
    # every cell is label-homogeneous, so strict labels never fail
    labels = mesh.element_labels()
    assert labels.shape[0] == len(mesh.elements)
    # every cell's averaged property is its exact label
    for element, leaf in zip(mesh.elements, mesh.leaves):
        assert element.element_property == float(leaf.array.ravel()[0])


def test_label_mode_with_max_size():
    image = label_image()
    quad = QTree(None, image_preprocess(image), 1, max_size=8, label_mode=True)
    mesh = QTreeMesh(quad)
    mesh.create_elements()
    assert max(e.element_type[2] for e in mesh.elements) <= 8.0
    assert (mesh.element_labels(strict=True) != 0).any()


def test_label_mode_rejects_crit():
    with pytest.raises(ValueError):
        QTree(None, np.zeros((8, 8), dtype=int), 3, label_mode=True)


def test_from_image_label_mode():
    image = label_image()
    mesh = QTreeMesh.from_image(image, label_mode=True)
    quad = QTree(None, image_preprocess(image), 1, label_mode=True)
    manual = QTreeMesh(quad)
    manual.create_elements()
    assert len(mesh.elements) == len(manual.elements)
    assert (mesh.element_labels() == manual.element_labels()).all()
