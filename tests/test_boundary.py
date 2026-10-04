"""Tests for the mesh-level boundary edges and nodes."""

import numpy as np

from qtreemesh import QTreeMesh
from qtreemesh.sbfem import SBFEMModel


def disk_mesh(size=32):
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size), dtype=int)
    image[((xx - size // 2) ** 2 + (yy - size // 2) ** 2) < size**2 // 16] = 3
    return QTreeMesh.from_image(image, crit=1)


def element_edge_set(mesh):
    edges = set()
    for element in mesh.elements:
        nodes = element.nodes_numbers
        n = len(nodes)
        for i in range(n):
            a, b = int(nodes[i]), int(nodes[(i + 1) % n])
            edges.add((min(a, b), max(a, b)))
    return edges


def test_boundary_edges_belong_to_one_element():
    mesh = disk_mesh()
    all_edges = element_edge_set(mesh)
    boundary = mesh.boundary_edges()
    assert all(edge in all_edges for edge in boundary)
    # every non-boundary edge is shared by exactly two elements
    counts = {}
    for element in mesh.elements:
        nodes = element.nodes_numbers
        n = len(nodes)
        for i in range(n):
            a, b = int(nodes[i]), int(nodes[(i + 1) % n])
            key = (min(a, b), max(a, b))
            counts[key] = counts.get(key, 0) + 1
    for edge in boundary:
        assert counts[edge] == 1
    for edge, count in counts.items():
        if edge not in set(boundary):
            assert count == 2


def test_boundary_nodes_are_the_boundary_edge_endpoints():
    mesh = disk_mesh()
    nodes = mesh.boundary_nodes()
    assert nodes == sorted(set(nodes))
    assert set(nodes) == {n for edge in mesh.boundary_edges() for n in edge}


def test_boundary_of_single_element_mesh():
    mesh = QTreeMesh.from_image(np.zeros((8, 8), dtype=int))
    assert len(mesh.elements) == 1
    assert len(mesh.boundary_edges()) == 4
    assert mesh.boundary_nodes() == [1, 2, 3, 4]


def test_sbfem_model_matches_mesh_boundary():
    mesh = disk_mesh()
    model = SBFEMModel(mesh, moduli=1.0, nu=0.4)
    assert model.boundary_edges() == mesh.boundary_edges()
    # sanity: boundary nodes all lie on the outer rectangle of the mesh
    xy = mesh.nodes[np.array(mesh.boundary_nodes()) - 1]
    pad = mesh.quad_tree.array.shape[0]
    assert np.all(
        (np.isclose(xy[:, 0], 0.0))
        | (np.isclose(xy[:, 0], pad))
        | (np.isclose(xy[:, 1], 0.0))
        | (np.isclose(xy[:, 1], pad))
    )
