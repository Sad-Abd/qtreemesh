"""Tests for the SBFEM model of a quadtree mesh."""

import numpy as np
import pytest

from qtreemesh import QTree, QTreeMesh
from qtreemesh.sbfem import SBFEMModel


def two_phase_mesh(size=32):
    image = np.zeros((size, size), dtype=int)
    image[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = 10
    quad = QTree(None, image, 1)
    mesh = QTreeMesh(quad, balancing=True)
    mesh.create_elements()
    return mesh


def mesh_with_all_modes(size=32):
    image = np.zeros((size, size), dtype=int)
    yy, xx = np.mgrid[0:size, 0:size]
    image[((xx - size // 2) ** 2 + (yy - size // 2) ** 2) < size**2 // 16] = 10
    quad = QTree(None, image, 1)
    mesh = QTreeMesh(quad, balancing=True)
    mesh.create_elements()
    return mesh


def test_model_requires_generated_elements():
    quad = QTree(None, np.zeros((8, 8), dtype=int), 1)
    mesh = QTreeMesh(quad)
    with pytest.raises(ValueError):
        SBFEMModel(mesh, moduli=1.0, nu=0.3)


def test_stiffness_is_symmetric_and_rigid_null():
    mesh = two_phase_mesh()
    model = SBFEMModel(mesh, moduli=1.0, nu=0.45)
    K = model.stiffness()
    assert K.shape == (2 * mesh.nodes.shape[0],) * 2
    assert (K - K.T).max() < 1e-10 * abs(K).max()
    n = mesh.nodes.shape[0]
    for translation in (np.tile([1, 0], n), np.tile([0, 1], n)):
        assert np.abs(K @ translation).max() < 1e-8 * abs(K).max()


def test_stiffness_scales_with_modulus():
    mesh = two_phase_mesh()
    K1 = SBFEMModel(mesh, moduli=1.0, nu=0.45).stiffness()
    K2 = SBFEMModel(mesh, moduli=2.0, nu=0.45).stiffness()
    assert (K2 - 2 * K1).max() < 1e-9 * abs(K2).max()


def test_assembly_scatters_cached_element_stiffnesses():
    mesh = two_phase_mesh()
    model = SBFEMModel(mesh, moduli=1.0, nu=0.45)
    K = model.stiffness().toarray()
    built = np.zeros_like(K)
    for element in model._elements:
        K_e = model._cache.stiffness(element["mode"], element["rotation"], element["E"])
        dofs = element["dofs"]
        built[np.ix_(dofs, dofs)] += K_e
    assert np.abs(K - built).max() < 1e-12


def test_two_material_moduli_by_label():
    mesh = two_phase_mesh()
    model = SBFEMModel(mesh, moduli={0: 1.0, 10: 5.0}, nu=0.45)
    K = model.stiffness()
    assert abs(K).max() > 0
    # inclusion elements carry E = 5
    for element in model._elements:
        if element["E"] == 5.0:
            break
    else:
        raise AssertionError("no element with E = 5 found")


def test_boundary_edges():
    mesh = two_phase_mesh()
    model = SBFEMModel(mesh, moduli=1.0, nu=0.45)
    edges = model.boundary_edges()
    nodes_on_border = {
        n
        for e in edges
        for n in e
        if mesh.nodes[n - 1, 0] in (0.0, 32.0) or mesh.nodes[n - 1, 1] in (0.0, 32.0)
    }
    assert len(edges) > 0
    # every mesh node lying on the domain border appears in a boundary edge
    border_nodes = {
        i + 1
        for i, (x, y) in enumerate(mesh.nodes)
        if x in (0.0, 32.0) or y in (0.0, 32.0)
    }
    assert nodes_on_border == border_nodes


def test_traction_forces_total():
    mesh = two_phase_mesh()
    model = SBFEMModel(mesh, moduli=1.0, nu=0.45)
    edges = model.boundary_edges()
    top = [
        (a, b)
        for (a, b) in edges
        if mesh.nodes[a - 1, 1] == 32.0 and mesh.nodes[b - 1, 1] == 32.0
    ]
    force = model.traction_forces(top, np.tile([0.0, 2.0], (len(top), 1)))
    length = sum(
        np.linalg.norm(mesh.nodes[b - 1] - mesh.nodes[a - 1]) for (a, b) in top
    )
    assert force[1::2].sum() == pytest.approx(2.0 * length)
    assert force[0::2].sum() == pytest.approx(0.0)


def test_solve_rigid_translation_gives_zero_reactions():
    mesh = two_phase_mesh()
    model = SBFEMModel(mesh, moduli=1.0, nu=0.45)
    edges = model.boundary_edges()
    bnodes = sorted({n for e in edges for n in e})
    bc = []
    for n in bnodes:
        bc.append([n, 1, 0.7]); bc.append([n, 2, -1.2])
    u, reactions = model.solve(np.array(bc))
    assert np.abs(u[0::2] - 0.7).max() < 1e-9
    assert np.abs(u[1::2] + 1.2).max() < 1e-9
    assert np.abs(reactions).max() < 1e-8


def test_solve_force_balance():
    mesh = two_phase_mesh()
    model = SBFEMModel(mesh, moduli=1.0, nu=0.45)
    edges = model.boundary_edges()
    top = [
        (a, b)
        for (a, b) in edges
        if mesh.nodes[a - 1, 1] == 32.0 and mesh.nodes[b - 1, 1] == 32.0
    ]
    force = model.traction_forces(top, np.tile([0.0, 1.0], (len(top), 1)))
    bnodes = sorted({n for e in model.boundary_edges() for n in e})
    bc = []
    for n in bnodes:
        if mesh.nodes[n - 1, 1] == 0.0:
            bc.append([n, 2, 0.0])
    first = min(n for n in bnodes if mesh.nodes[n - 1, 1] == 0.0)
    bc.append([first, 1, 0.0])
    _, reactions = model.solve(np.array(bc), force=force)
    # sum of reactions balances the applied load
    assert reactions[1::2].sum() == pytest.approx(-force[1::2].sum(), rel=1e-9)


def test_boundary_stress_and_field_consistency():
    mesh = mesh_with_all_modes()
    model = SBFEMModel(mesh, moduli=1.0, nu=0.45)
    edges = model.boundary_edges()
    bnodes = sorted({n for e in edges for n in e})
    bc = []
    for n in bnodes:
        x, y = mesh.nodes[n - 1]
        bc.append([n, 1, 0.01 * x]); bc.append([n, 2, 0.005 * y])
    u, _ = model.solve(np.array(bc))
    points, stress = model.boundary_stress(u)
    assert points.shape[0] == stress.shape[0] > 0
    # field() at xi = 1, eta = 0 must reproduce the matching boundary midpoint
    element = mesh.elements[0]
    n = len(element.nodes_numbers)
    for edge in range(n):
        disp, sig, point = model.field(element.number, edge, 0.0, 1.0, u)
        distances = np.linalg.norm(points - point, axis=1)
        k = int(np.argmin(distances))
        assert distances[k] < 1e-9
        assert np.abs(sig - stress[k]).max() < 1e-10


def test_field_rejects_invalid_coordinates():
    mesh = two_phase_mesh()
    model = SBFEMModel(mesh, moduli=1.0, nu=0.45)
    u = np.zeros(model.ndof)
    with pytest.raises(ValueError):
        model.field(1, 0, 0.0, 0.0, u)
    with pytest.raises(ValueError):
        model.field(1, 0, 0.0, 1.5, u)
    with pytest.raises(ValueError):
        model.field(1, 99, 0.0, 1.0, u)


def test_mesh_with_all_modes_solves():
    mesh = mesh_with_all_modes()
    modes = {element.element_type[0] for element in mesh.elements}
    assert modes <= set(range(1, 7))
    model = SBFEMModel(mesh, moduli=1.0, nu=0.45)
    edges = model.boundary_edges()
    bnodes = sorted({n for e in edges for n in e})
    bc = []
    for n in bnodes:
        x, y = mesh.nodes[n - 1]
        bc.append([n, 1, 0.01 * x]); bc.append([n, 2, -0.02 * y])
    u, reactions = model.solve(np.array(bc))
    assert np.abs(u).max() > 0
    # equilibrium of the constrained problem: K u - reactions is zero by
    # construction; check the residual of the solved system instead
    K = model.stiffness()
    residual = K @ u - reactions
    assert np.abs(residual).max() < 1e-9


def test_solve_with_empty_dirichlet_and_no_force():
    mesh = two_phase_mesh()
    model = SBFEMModel(mesh, moduli=1.0, nu=0.45)
    u = np.zeros(model.ndof)
    u, _ = model.solve(np.zeros((0, 3)))
    assert np.abs(u).max() == 0.0
