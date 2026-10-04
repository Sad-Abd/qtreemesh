"""Tests for the hanging-node constraint alternative to triangulation."""

import numpy as np

from qtreemesh import QTreeMesh


def disk_mesh(size=32):
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size), dtype=int)
    image[((xx - size // 2) ** 2 + (yy - size // 2) ** 2) < size**2 // 16] = 3
    return QTreeMesh.from_image(image, crit=1)


def test_every_element_is_a_corner_quad():
    mesh = disk_mesh()
    nodes, elements, properties, constraints = mesh.constrained_quads()
    assert nodes.shape == mesh.nodes.shape
    assert len(elements) == len(mesh.elements) == len(properties)
    for element, quad in zip(mesh.elements, elements):
        assert len(quad) == 4
        assert quad == element.corner_numbers
        assert all(1 <= n <= nodes.shape[0] for n in quad)


def test_hanging_nodes_are_edge_midpoints():
    mesh = disk_mesh()
    nodes, elements, properties, constraints = mesh.constrained_quads()
    assert constraints
    seen = set()
    for hanging, corner_a, corner_b in constraints:
        assert hanging not in seen
        seen.add(hanging)
        expected = (
            mesh.nodes[corner_a - 1] + mesh.nodes[corner_b - 1]
        ) / 2.0
        assert np.allclose(mesh.nodes[hanging - 1], expected)


def test_constraints_cover_every_hanging_node():
    mesh = disk_mesh()
    _, _, _, constraints = mesh.constrained_quads()
    total_hanging = sum(len(e.nodes_numbers) - 4 for e in mesh.elements)
    assert len(constraints) == total_hanging


def test_quad_elements_match_mesh_geometry():
    mesh = disk_mesh()
    _, elements, _, _ = mesh.constrained_quads()
    for element, quad in zip(mesh.elements, elements):
        corners = mesh.nodes[np.array(quad) - 1]
        cell = np.asarray(element.nodes_coordinates, dtype=float)
        assert np.isclose(corners[:, 0].min(), cell[:, 0].min())
        assert np.isclose(corners[:, 0].max(), cell[:, 0].max())
        assert np.isclose(corners[:, 1].min(), cell[:, 1].min())
        assert np.isclose(corners[:, 1].max(), cell[:, 1].max())


def test_uniform_mesh_has_no_constraints():
    mesh = QTreeMesh.from_image(np.zeros((8, 8), dtype=int))
    _, elements, properties, constraints = mesh.constrained_quads()
    assert len(elements) == 1
    assert properties == [0.0]
    assert constraints == []
