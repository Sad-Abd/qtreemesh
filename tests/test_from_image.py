"""Tests for the one-step mesh constructor."""

import numpy as np

from qtreemesh import QTree, QTreeMesh, image_preprocess


def disk_image(size=48):
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size), dtype=int)
    image[((xx - size // 2) ** 2 + (yy - size // 2) ** 2) < size**2 // 16] = 7
    return image


def test_from_image_matches_manual_pipeline():
    image = disk_image()
    mesh = QTreeMesh.from_image(image, crit=1)
    quad = QTree(None, image_preprocess(image), 1)
    manual = QTreeMesh(quad)
    manual.create_elements()
    assert mesh.nodes.shape == manual.nodes.shape
    assert np.allclose(mesh.nodes, manual.nodes)
    assert len(mesh.elements) == len(manual.elements)
    for a, b in zip(mesh.elements, manual.elements):
        assert a.number == b.number
        assert a.nodes_numbers == b.nodes_numbers
        assert a.element_type == b.element_type
        assert a.element_property == b.element_property


def test_from_image_respects_max_size():
    image = disk_image()
    mesh = QTreeMesh.from_image(image, crit=1, max_size=8)
    assert max(e.element_type[2] for e in mesh.elements) <= 8.0


def test_from_image_respects_scale():
    image = disk_image()
    mesh = QTreeMesh.from_image(image, crit=1, scale=2.0)
    assert mesh.nodes[:, 0].max() == 2.0 * image_preprocess(image).shape[1]


def test_from_image_without_balancing():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 4, size=(16, 16))
    mesh = QTreeMesh.from_image(image, crit=0, balancing=False)
    quad = QTree(None, image_preprocess(image), 0)
    manual = QTreeMesh(quad, balancing=False)
    manual.create_elements()
    assert len(mesh.elements) == len(manual.elements)
    for a, b in zip(mesh.elements, manual.elements):
        assert a.element_type == b.element_type


def test_from_image_then_trim():
    image = disk_image()
    mesh = QTreeMesh.from_image(image, crit=1)
    before = len(mesh.elements)
    mesh.trim_padding(image.shape[0], image.shape[1])
    assert 0 < len(mesh.elements) < before
    assert mesh.content_shape == (48, 48)
