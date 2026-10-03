"""Tests for the pixel → element lookup and exact per-cell labels."""

import numpy as np

from qtreemesh import QTree, QTreeMesh, image_preprocess


def label_image():
    """Three distinct regions (values spaced well beyond crit) on a background.

    Values must differ by more than crit, else adjacent regions are not split
    and cells end up inhomogeneous."""
    image = np.zeros((64, 64), dtype=int)
    image[8:32, 8:32] = 10
    image[40:56, 40:60] = 20
    return image


def label_mesh(image=None, **kwargs):
    image = label_image() if image is None else image
    quad = QTree(None, image, 1, **kwargs)
    mesh = QTreeMesh(quad, balancing=True)
    mesh.create_elements()
    return mesh


def test_pixel_to_element_covers_every_pixel():
    mesh = label_mesh()
    pixel_elem = mesh.pixel_to_element()
    assert pixel_elem.shape == label_image().shape
    assert (pixel_elem > 0).all()


def test_pixel_to_element_numbers_match_elements():
    mesh = label_mesh()
    pixel_elem = mesh.pixel_to_element()
    assert set(np.unique(pixel_elem)) == {e.number for e in mesh.elements}


def test_pixel_to_element_pixels_form_the_cell_box():
    mesh = label_mesh()
    pixel_elem = mesh.pixel_to_element()
    for element in mesh.elements:
        xy = np.asarray(element.nodes_coordinates, dtype=float)
        size = int(round(xy.max(axis=0)[0] - xy.min(axis=0)[0]))
        assert (pixel_elem == element.number).sum() == size * size


def test_pixel_to_element_flips_rows_top_to_bottom():
    mesh = label_mesh()
    pixel_elem = mesh.pixel_to_element()
    n = 64
    # the element covering the top-left pixel must touch x = 0 and y = n
    corner = mesh.elements[pixel_elem[0, 0] - 1]
    xy = np.asarray(corner.nodes_coordinates, dtype=float)
    assert xy.min(axis=0)[0] == 0
    assert xy.max(axis=0)[1] == n


def test_element_labels_are_exact():
    mesh = label_mesh()
    image = label_image()
    pixel_elem = mesh.pixel_to_element()
    labels = mesh.element_labels()
    assert len(labels) == len(mesh.elements)
    for element, label in zip(mesh.elements, labels):
        under = image[pixel_elem == element.number]
        assert under.min() == under.max() == label


def test_element_labels_strict_raises_when_a_cell_spans_labels():
    quad = QTree(None, label_image(), 200)  # no refinement: one merged cell
    mesh = QTreeMesh(quad, balancing=True)
    mesh.create_elements()
    try:
        mesh.element_labels()
    except ValueError as error:
        assert "span multiple intensities" in str(error)
    else:
        raise AssertionError("expected ValueError for inhomogeneous cells")


def test_element_labels_non_strict_falls_back_to_minimum():
    image = label_image()
    quad = QTree(None, image, 200)
    mesh = QTreeMesh(quad, balancing=True)
    mesh.create_elements()
    labels = mesh.element_labels(strict=False)
    assert list(labels) == [image.min()]


def test_pixel_to_element_raises_when_pixels_are_uncovered():
    mesh = label_mesh()
    mesh.elements.pop()  # simulate an incomplete element list
    try:
        mesh.pixel_to_element()
    except RuntimeError as error:
        assert "not covered" in str(error)
    else:
        raise AssertionError("expected RuntimeError for uncovered pixels")


def test_pixel_to_element_with_scale():
    mesh = label_mesh(scale=2.0)
    pixel_elem = mesh.pixel_to_element()
    assert pixel_elem.shape == label_image().shape
    assert (pixel_elem > 0).all()
    assert set(np.unique(pixel_elem)) == {e.number for e in mesh.elements}


def test_pixel_to_element_covers_padding_of_padded_images():
    image = np.zeros((100, 60), dtype=int)
    image[20:50, 10:40] = 3
    padded = image_preprocess(image)
    quad = QTree(None, padded, 1)
    mesh = QTreeMesh(quad, balancing=True)
    mesh.create_elements()
    pixel_elem = mesh.pixel_to_element()
    assert pixel_elem.shape == padded.shape == (128, 128)
    assert (pixel_elem > 0).all()
