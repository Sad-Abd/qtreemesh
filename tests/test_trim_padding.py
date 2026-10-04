"""Tests for removing padded-region cells from a mesh."""

import numpy as np
import pytest

from qtreemesh import QTree, QTreeMesh, image_preprocess


def rect_mesh():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 3, size=(20, 37))
    quad = QTree(None, image_preprocess(image), 1)
    mesh = QTreeMesh(quad)
    mesh.create_elements()
    return mesh, image


def content_mesh():
    image = np.zeros((32, 32), dtype=int)
    image[:10, :10] = 5
    quad = QTree(None, image, 1)
    mesh = QTreeMesh(quad)
    mesh.create_elements()
    return mesh


def bbox(element):
    xy = np.asarray(element.nodes_coordinates, dtype=float)
    return xy.min(axis=0), xy.max(axis=0)


def intersects_content(element, rows, cols, padded_rows, scale=1.0):
    lo, hi = bbox(element)
    x_ok = lo[0] < cols * scale and hi[0] > 0.0
    y_ok = lo[1] < padded_rows * scale and hi[1] > (padded_rows - rows) * scale
    return x_ok and y_ok


def test_trim_removes_only_padding_cells():
    mesh, _ = rect_mesh()
    padded_rows = mesh.quad_tree.array.shape[0]
    before = list(mesh.elements)
    mesh.trim_padding(20, 37)
    assert mesh.content_shape == (20, 37)
    assert 0 < len(mesh.elements) < len(before)
    for element in mesh.elements:
        assert intersects_content(element, 20, 37, padded_rows)
    assert set(e.number for e in mesh.elements) <= set(e.number for e in before)


def test_trim_keeps_cells_that_straddle_content():
    quad = QTree(None, np.zeros((32, 32), dtype=int), 1)
    mesh = QTreeMesh(quad)
    mesh.create_elements()
    mesh.trim_padding(10, 10)
    assert len(mesh.elements) == 1
    assert (mesh.pixel_to_element() > 0).all()


def test_pixel_to_element_after_trim():
    mesh, _ = rect_mesh()
    mesh.trim_padding(20, 37)
    pixel_elem = mesh.pixel_to_element()
    assert (pixel_elem[:20, :37] > 0).all()
    assert (pixel_elem == 0).any()
    kept = np.array([e.number for e in mesh.elements])
    assert np.isin(pixel_elem[pixel_elem > 0], kept).all()


def test_element_labels_aligned_after_trim():
    mesh = content_mesh()
    mesh.trim_padding(10, 10)
    labels = mesh.element_labels()
    assert labels.shape[0] == len(mesh.elements)
    assert (labels == 5).all()
    for element in mesh.elements:
        assert intersects_content(element, 10, 10, 32)


def test_trim_without_padding_change():
    image = np.zeros((32, 32), dtype=int)
    image[5:15, 5:15] = 2
    quad = QTree(None, image, 1)
    mesh = QTreeMesh(quad)
    mesh.create_elements()
    before = len(mesh.elements)
    mesh.trim_padding(32, 32)
    assert len(mesh.elements) == before
    assert (mesh.pixel_to_element() > 0).all()


def test_trim_invalid_shape():
    mesh = content_mesh()
    with pytest.raises(ValueError):
        mesh.trim_padding(0, 10)
    with pytest.raises(ValueError):
        mesh.trim_padding(10, 100)


def test_trim_twice_and_covered_pad_pixels():
    mesh = content_mesh()
    mesh.trim_padding(10, 10)
    mesh.trim_padding(10, 10)
    assert mesh.content_shape == (10, 10)
    pixel_elem = mesh.pixel_to_element()
    assert (pixel_elem[:10, :10] > 0).all()
    assert (pixel_elem == 0).any()
