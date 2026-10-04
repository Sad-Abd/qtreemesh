"""Tests for the exact SBFEM coefficient matrices and pattern geometry."""

import numpy as np
import pytest

from qtreemesh.sbfem import (
    CANONICAL_MODES,
    canonical_polygon,
    edge_operators,
    isotropic_tangent,
    polygon_E0E1E2,
    transform,
)


def gauss_reference_E(xy, A, n=40):
    """Independent high-order Gauss evaluation of the E-integrals."""
    pts, wts = np.polynomial.legendre.leggauss(n)
    nd = 2 * xy.shape[0]
    E0 = np.zeros((nd, nd))
    E1 = np.zeros((nd, nd))
    E2 = np.zeros((nd, nd))
    for i in range(xy.shape[0]):
        p, q = xy[i], xy[(i + 1) % xy.shape[0]]
        d = q - p
        m = 0.5 * (p + q)
        Jb = 0.5 * (m[0] * d[1] - m[1] * d[0])
        b1 = (1.0 / Jb) * np.array(
            [[d[1] / 2, 0], [0, -d[0] / 2], [-d[0] / 2, 0], [0, d[1] / 2]]
        )
        dm = [2 * i, 2 * i + 1, 2 * ((i + 1) % xy.shape[0]), 2 * ((i + 1) % xy.shape[0]) + 1]
        for w, eta in zip(wts, pts):
            b2 = (1.0 / Jb) * np.array(
                [
                    [-(m[1] + d[1] * eta / 2), 0],
                    [0, m[0] + d[0] * eta / 2],
                    [m[0] + d[0] * eta / 2, 0],
                    [0, -(m[1] + d[1] * eta / 2)],
                ]
            )
            N1 = (1 - eta) / 2
            N2 = (1 + eta) / 2
            B1 = b1 @ np.array([[N1, 0, N2, 0], [0, N1, 0, N2]])
            B2 = b2 @ np.array([[-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]])
            E0[np.ix_(dm, dm)] += w * Jb * B1.T @ A @ B1
            E1[np.ix_(dm, dm)] += w * Jb * B2.T @ A @ B1
            E2[np.ix_(dm, dm)] += w * Jb * B2.T @ A @ B2
    return E0, E1, E2


def random_star_polygon(rng, n=7):
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    radii = rng.uniform(0.5, 1.5, n)
    xy = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return xy - xy.mean(axis=0)


@pytest.mark.parametrize("mode", CANONICAL_MODES)
def test_closed_form_matches_gauss_reference(mode):
    A = isotropic_tangent(1.0, 0.3, "plane_strain")
    xy = canonical_polygon(mode)
    for a, b in zip(polygon_E0E1E2(xy, A), gauss_reference_E(xy, A)):
        assert np.abs(a - b).max() < 1e-12


def test_closed_form_matches_gauss_reference_random_polygon():
    rng = np.random.default_rng(7)
    A = isotropic_tangent(2.0, 0.1, "plane_stress")
    xy = random_star_polygon(rng)
    for a, b in zip(polygon_E0E1E2(xy, A), gauss_reference_E(xy, A)):
        assert np.abs(a - b).max() < 1e-12


def test_E_matrices_linear_in_material():
    rng = np.random.default_rng(3)
    xy = random_star_polygon(rng)
    lambda_part = np.array([[0.0, 1.0, 0, 0], [1.0, 0.0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    mu_part = np.array([[2.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 1.0, 1.0], [0, 0, 1.0, 1.0]])
    lam, mu = 0.4, 1.7
    baked = [lam * e + mu * m for e, m in zip(
        polygon_E0E1E2(xy, lambda_part), polygon_E0E1E2(xy, mu_part))]
    direct = polygon_E0E1E2(xy, lam * lambda_part + mu * mu_part)
    for a, b in zip(baked, direct):
        assert np.abs(a - b).max() < 1e-13


def test_translations_are_null_vectors():
    """E1^T and E2 annihilate the rigid translations (E0 and E1 do not)."""
    for mode in CANONICAL_MODES:
        xy = canonical_polygon(mode)
        A = isotropic_tangent(1.0, 0.45, "plane_strain")
        E0, E1, E2 = polygon_E0E1E2(xy, A)
        n = xy.shape[0]
        for translation in (np.tile([1, 0], n), np.tile([0, 1], n)):
            assert np.abs(E2 @ translation).max() < 1e-12
            assert np.abs(E1.T @ translation).max() < 1e-12


def test_isotropic_tangent_plane_stress_strain():
    E, nu = 2.5, 0.25
    a_strain = isotropic_tangent(E, nu, "plane_strain")
    a_stress = isotropic_tangent(E, nu, "plane_stress")
    assert a_strain[0, 0] == pytest.approx(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu)))
    assert a_stress[0, 0] == pytest.approx(E / (1 - nu**2))
    with pytest.raises(ValueError):
        isotropic_tangent(E, nu, "unknown")


def test_edge_operators_recover_constant_strain():
    """The operators applied to a linear field give its constant gradient.

    For u(xi, eta) = xi * N(eta) u_boundary (a linear field), the strain is
    eps = B1 u'(1) + B2 u(1) with u'(1) = u(1) = the nodal values."""
    xy = canonical_polygon(1)
    F = np.array([[0.3, 0.2], [-0.2, 0.4]])
    u = np.array([F @ p for p in xy]).reshape(-1)  # interleaved [ux1, uy1, ...]
    for i in range(4):
        j = (i + 1) % 4
        B1, B2 = edge_operators(xy[i], xy[j], 0.3)
        dm = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        eps = B1 @ u[dm] + B2 @ u[dm]
        assert eps[0] == pytest.approx(F[0, 0], abs=1e-12)
        assert eps[1] == pytest.approx(F[1, 1], abs=1e-12)
        assert eps[2] == pytest.approx(F[0, 1], abs=1e-12)
        assert eps[3] == pytest.approx(F[1, 0], abs=1e-12)


def test_degenerate_edge_raises():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    A = isotropic_tangent(1.0, 0.3)
    with pytest.raises(ValueError):
        polygon_E0E1E2(xy, A)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        canonical_polygon(7)


@pytest.mark.parametrize(
    "mode,rotation",
    [(1, 90), (6, 90), (4, 180), (2, 45), (3, 30), (5, 45)],
)
def test_invalid_rotations_raise(mode, rotation):
    with pytest.raises(ValueError):
        transform(mode, rotation)


def test_transform_is_orthogonal():
    for mode in CANONICAL_MODES:
        n = canonical_polygon(mode).shape[0]
        for rotation in (0, 90, 180, 270):
            try:
                T = transform(mode, rotation)
            except ValueError:
                continue
            assert np.abs(T @ T.T - np.eye(2 * n)).max() < 1e-13


def test_unknown_mode_in_transform_raises():
    with pytest.raises(ValueError):
        transform(9, 0)


@pytest.mark.parametrize(
    "E,nu,formulation",
    [
        (1.0, 0.5, "plane_strain"),
        (1.0, -1.0, "plane_strain"),
        (1.0, 0.7, "plane_strain"),
        (1.0, 1.0, "plane_stress"),
        (1.0, -1.5, "plane_stress"),
        (0.0, 0.3, "plane_strain"),
        (-2.0, 0.3, "plane_stress"),
        (np.inf, 0.3, "plane_strain"),
        (np.nan, 0.3, "plane_strain"),
        (1.0, np.nan, "plane_strain"),
    ],
)
def test_isotropic_tangent_rejects_invalid_parameters(E, nu, formulation):
    with pytest.raises(ValueError):
        isotropic_tangent(E, nu, formulation)


def test_isotropic_tangent_accepts_the_open_poisson_range():
    isotropic_tangent(1.0, -0.99, "plane_strain")
    isotropic_tangent(1.0, 0.4999999, "plane_strain")
    isotropic_tangent(1.0, 0.99, "plane_stress")
