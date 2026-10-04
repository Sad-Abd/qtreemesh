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


# --- stress and interior-field recovery against analytic solutions ---------


def mesh_all_patterns():
    """A mesh containing every (mode, rotation) combination of the patterns."""
    rng = np.random.default_rng(2)
    n = 128
    image = np.zeros((n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    for _ in range(6):
        cx, cy = rng.uniform(10, 118, 2)
        r = rng.uniform(3, 15)
        image[(xx - cx) ** 2 + (yy - cy) ** 2 < r**2] = 255
    mesh = QTreeMesh(QTree(None, image, 1), balancing=True)
    mesh.create_elements()
    return mesh


def border_nodes(model):
    return sorted({n for edge in model.boundary_edges() for n in edge})


def linear_field_solution(model, mesh, G):
    """Solve with the boundary displacements of the linear field u = G x."""
    bc = [
        [n, c + 1, (G @ mesh.nodes[n - 1])[c]]
        for n in border_nodes(model)
        for c in (0, 1)
    ]
    u, _ = model.solve(np.array(bc))
    return u


def uniform_stress(G, E, nu, formulation="plane_strain"):
    """[sxx, syy, sxy] of the uniform strain G for an isotropic material."""
    if formulation == "plane_strain":
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    else:
        lmbda = E * nu / (1 - nu**2)
    mu = E / (2 * (1 + nu))
    exx, eyy, exy = G[0, 0], G[1, 1], 0.5 * (G[0, 1] + G[1, 0])
    trace = exx + eyy
    return np.array(
        [lmbda * trace + 2 * mu * exx, lmbda * trace + 2 * mu * eyy, 2 * mu * exy]
    )


G_UNIFORM = np.array([[0.004, 0.0015], [-0.0005, -0.002]])


def test_test_mesh_has_every_pattern_and_rotation():
    mesh = mesh_all_patterns()
    combinations = {tuple(e.element_type[:2]) for e in mesh.elements}
    expected = {(1, 0), (4, 0), (4, 90), (6, 0)}
    expected |= {(m, r) for m in (2, 3, 5) for r in (0, 90, 180, 270)}
    assert combinations == expected


@pytest.mark.parametrize("modulus", [1.0, 5.0])
def test_boundary_stress_of_uniform_strain_includes_modulus(modulus):
    mesh = mesh_all_patterns()
    model = SBFEMModel(mesh, moduli=modulus, nu=0.3)
    u = linear_field_solution(model, mesh, G_UNIFORM)
    _, stress = model.boundary_stress(u)
    exact = uniform_stress(G_UNIFORM, modulus, 0.3)
    assert np.abs(stress - exact).max() < 1e-12 * np.abs(exact).max()


def test_boundary_stress_rows_follow_the_element_edges():
    mesh = mesh_all_patterns()
    model = SBFEMModel(mesh, moduli=1.0, nu=0.3)
    u = linear_field_solution(model, mesh, G_UNIFORM)
    points, _ = model.boundary_stress(u)
    row = 0
    for element in mesh.elements:
        xy = np.asarray(element.nodes_coordinates, dtype=float)
        n = len(xy)
        for edge in range(n):
            midpoint = 0.5 * (xy[edge] + xy[(edge + 1) % n])
            assert np.linalg.norm(points[row] - midpoint) < 1e-9
            row += 1
    assert row == points.shape[0]


def test_field_edge_index_follows_the_element_node_list():
    """edge i of field() runs from node i to node i+1 of the element, also
    for rotated cells."""
    mesh = mesh_all_patterns()
    model = SBFEMModel(mesh, moduli=1.0, nu=0.3)
    u = linear_field_solution(model, mesh, G_UNIFORM)
    for element in mesh.elements:
        xy = np.asarray(element.nodes_coordinates, dtype=float)
        n = len(xy)
        for edge in range(n):
            _, _, start = model.field(element.number, edge, -1.0, 1.0, u)
            _, _, end = model.field(element.number, edge, 1.0, 1.0, u)
            assert np.linalg.norm(start - xy[edge]) < 1e-9
            assert np.linalg.norm(end - xy[(edge + 1) % n]) < 1e-9


@pytest.mark.parametrize("modulus", [1.0, 5.0])
@pytest.mark.parametrize("xi", [1.0, 0.5, 0.1, 0.01])
def test_field_reproduces_uniform_strain_inside_elements(modulus, xi):
    """Displacement and stress at interior points of every kind of cell match
    the analytic linear field, including the modulus."""
    mesh = mesh_all_patterns()
    model = SBFEMModel(mesh, moduli=modulus, nu=0.3)
    u = linear_field_solution(model, mesh, G_UNIFORM)
    exact = uniform_stress(G_UNIFORM, modulus, 0.3)
    seen = set()
    for element in mesh.elements:
        key = tuple(element.element_type[:2])
        if key in seen:
            continue
        seen.add(key)
        for edge in range(len(element.nodes_numbers)):
            for eta in (-0.8, 0.0, 0.6):
                disp, sigma, point = model.field(element.number, edge, eta, xi, u)
                assert np.abs(disp - G_UNIFORM @ point).max() < 1e-12
                assert np.abs(sigma - exact).max() < 1e-10 * np.abs(exact).max()
    assert len(seen) == 16


def test_field_at_boundary_matches_boundary_stress():
    mesh = mesh_all_patterns()
    model = SBFEMModel(mesh, moduli=3.0, nu=0.4)
    rng = np.random.default_rng(5)
    u = rng.standard_normal(model.ndof) * 1e-3
    points, stress = model.boundary_stress(u)
    row = 0
    for element in mesh.elements[:60]:
        for edge in range(len(element.nodes_numbers)):
            _, sigma, point = model.field(element.number, edge, 0.0, 1.0, u)
            k = row + edge
            assert np.linalg.norm(points[k] - point) < 1e-9
            assert np.abs(sigma - stress[k]).max() < 1e-12
        row += len(element.nodes_numbers)


def test_stress_of_parallel_layers_scales_with_each_modulus():
    """Plane stress, parallel vertical interfaces, uniaxial strain in y: the
    solution is exactly uniform in strain with sigma_yy = E_cell * e0."""
    n = 64
    image = np.zeros((n, n), dtype=int)
    image[:, int(0.2 * n) : int(0.3 * n)] = 20
    image[:, n // 2 :] = 10
    mesh = QTreeMesh(QTree(None, image, 1), balancing=True)
    mesh.create_elements()
    moduli = {0: 1.0, 20: 3.0, 10: 10.0}
    nu, e0 = 0.3, 0.003
    model = SBFEMModel(mesh, moduli=moduli, nu=nu, formulation="plane_stress")
    G = np.diag([-nu * e0, e0])
    u = linear_field_solution(model, mesh, G)
    assert np.abs(u - (mesh.nodes @ G.T).ravel()).max() < 1e-12
    points, stress = model.boundary_stress(u)
    row = 0
    for element, number in zip(model._elements, mesh.elements):
        for _ in range(len(number.nodes_numbers)):
            assert stress[row, 1] == pytest.approx(element["E"] * e0, rel=1e-9)
            assert stress[row, 0] == pytest.approx(0.0, abs=1e-12)
            row += 1


def test_uniaxial_traction_patch_recovers_the_applied_stress():
    mesh = mesh_all_patterns()
    E, nu, t = 3.0, 0.3, 0.02
    model = SBFEMModel(mesh, moduli=E, nu=nu)
    top = mesh.nodes[:, 1].max()
    edges = model.boundary_edges()
    loaded = [
        (a, b)
        for (a, b) in edges
        if mesh.nodes[a - 1, 1] == top and mesh.nodes[b - 1, 1] == top
    ]
    force = model.traction_forces(loaded, np.tile([0.0, t], (len(loaded), 1)))
    nodes = border_nodes(model)
    bc = [[k, 1, 0.0] for k in nodes if mesh.nodes[k - 1, 0] == 0.0]
    bc += [[k, 2, 0.0] for k in nodes if mesh.nodes[k - 1, 1] == 0.0]
    u, _ = model.solve(np.array(bc), force=force)
    eyy = t * (1 - nu**2) / E
    exx = -nu * (1 + nu) * t / E
    exact = np.column_stack([exx * mesh.nodes[:, 0], eyy * mesh.nodes[:, 1]]).ravel()
    assert np.abs(u - exact).max() < 1e-12
    _, stress = model.boundary_stress(u)
    assert np.abs(stress[:, 1] - t).max() < 1e-10
    assert np.abs(stress[:, 0]).max() < 1e-10
