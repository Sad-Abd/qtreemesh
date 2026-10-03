"""Regression tests for QTreeMesh.labeling().

The labeling is the node-numbering contract of the whole package: downstream
code (refactor_edge, quad_treatment, vtk_export) indexes nodes as number - 1,
so any change must preserve the exact 1-based, first-seen numbering. The
reference implementation here is the original linear-search algorithm, kept
frozen on purpose.
"""

import numpy as np

from qtreemesh import QTree, QTreeMesh


def reference_labeling(mesh):
    """The pre-port O(N^2) labeling, kept as the behavioral reference."""
    mesh.nodes = np.array([[+np.inf, -np.inf]])
    label = 1

    for leaf in mesh.leaves:
        leaf.edge_points_numbers = []
        leaf.nodes_coordinate = []
        leaf.nodes_coordinate.append(np.array(leaf.bottom_left_corner.xy_coord))
        leaf.nodes_coordinate.append(
            np.array((leaf.top_right_corner.x_coord, leaf.bottom_left_corner.y_coord))
        )
        leaf.nodes_coordinate.append(np.array(leaf.top_right_corner.xy_coord))
        leaf.nodes_coordinate.append(
            np.array((leaf.bottom_left_corner.x_coord, leaf.top_right_corner.y_coord))
        )
        for node in leaf.nodes_coordinate:
            if any(np.equal(mesh.nodes, node).all(1)):
                leaf.edge_points_numbers.append(
                    np.where(np.equal(mesh.nodes, node).all(1) == True)[0][0]
                )
            else:
                mesh.nodes = np.r_[mesh.nodes, [node]]
                leaf.edge_points_numbers.append(mesh.nodes.shape[0] - 1)

        leaf.cell_number = label
        label += 1
    mesh.nodes = mesh.nodes[1:, :]


def sample_image():
    """A 64x64 image (gradient plus an ellipse) that refines to many cells."""
    y, x = np.mgrid[0:64, 0:64]
    image = 100 + (x // 8).astype(int)
    ellipse = ((x - 33) / 14.0) ** 2 + ((y - 30) / 7.0) ** 2 <= 1
    image[ellipse] = 25
    return image


def test_labeling_matches_reference(monkeypatch):
    """create_elements() output is identical to the pre-port labeling."""
    image = sample_image()
    monkeypatch.setattr(QTreeMesh, "labeling", reference_labeling)
    ref = QTreeMesh(QTree(None, image, 1), balancing=True)
    ref.create_elements()
    monkeypatch.undo()
    new = QTreeMesh(QTree(None, image, 1), balancing=True)
    new.create_elements()

    assert np.array_equal(ref.nodes, new.nodes)
    assert len(ref.elements) == len(new.elements)
    for ref_elem, new_elem in zip(ref.elements, new.elements):
        assert list(ref_elem.nodes_numbers) == list(new_elem.nodes_numbers)
        assert np.array_equal(
            np.array(ref_elem.nodes_coordinates),
            np.array(new_elem.nodes_coordinates),
        )
        assert list(ref_elem.element_type) == list(new_elem.element_type)


def test_labeling_invariants():
    """Node numbers are 1-based, dense, and point at the stored coordinates."""
    image = sample_image()
    mesh = QTreeMesh(QTree(None, image, 1), balancing=True)
    mesh.labeling()

    numbers = [n for leaf in mesh.leaves for n in leaf.edge_points_numbers]
    assert min(numbers) == 1
    assert max(numbers) == mesh.nodes.shape[0]
    assert len(set(numbers)) == mesh.nodes.shape[0]
    for leaf in mesh.leaves:
        assert leaf.cell_number >= 1
        for number, coord in zip(leaf.edge_points_numbers, leaf.nodes_coordinate):
            assert tuple(mesh.nodes[number - 1]) == tuple(coord)


def test_labeling_uniform_image():
    """A homogeneous image is a single cell with four nodes."""
    image = np.full((8, 8), 7)
    mesh = QTreeMesh(QTree(None, image, 1), balancing=True)
    mesh.create_elements()

    assert len(mesh.elements) == 1
    assert mesh.nodes.shape == (4, 2)
    assert mesh.elements[0].nodes_numbers == [1, 2, 3, 4]
    assert mesh.elements[0].number == 1
