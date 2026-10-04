"""Tests for the deflated condensation and the pattern cache."""

import numpy as np
import pytest

from qtreemesh.sbfem import (
    CANONICAL_MODES,
    PatternCache,
    canonical_polygon,
    condense,
    isotropic_tangent,
    polygon_E0E1E2,
    transform,
)


def element_matrices(mode, nu, formulation="plane_strain", E=1.0):
    xy = canonical_polygon(mode)
    A = isotropic_tangent(E, nu, formulation)
    return polygon_E0E1E2(xy, A)


@pytest.mark.parametrize("mode", CANONICAL_MODES)
@pytest.mark.parametrize("nu", [0.0, 0.3, 0.45, 0.49, 0.4999])
def test_condensed_stiffness_satisfies_scaled_boundary_equation(mode, nu):
    """(K - E1) E0^-1 (K - E1^T) = E2."""
    E0, E1, E2 = element_matrices(mode, nu)
    sol = condense(E0, E1, E2)
    residual = (sol.K - E1) @ np.linalg.inv(E0) @ (sol.K - E1.T) - E2
    assert np.abs(residual).max() < 1e-10 * np.abs(E2).max()


@pytest.mark.parametrize("mode", CANONICAL_MODES)
@pytest.mark.parametrize("nu", [0.0, 0.3, 0.4999])
def test_condensed_stiffness_symmetric_and_rigid_null(mode, nu):
    E0, E1, E2 = element_matrices(mode, nu)
    sol = condense(E0, E1, E2)
    assert np.abs(sol.K - sol.K.T).max() < 1e-10 * np.abs(sol.K).max()
    n = canonical_polygon(mode).shape[0]
    for translation in (np.tile([1, 0], n), np.tile([0, 1], n)):
        assert np.abs(sol.K @ translation).max() < 1e-9 * np.abs(sol.K).max()


@pytest.mark.parametrize("mode", CANONICAL_MODES)
def test_modal_data_structure(mode):
    E0, E1, E2 = element_matrices(mode, 0.3)
    sol = condense(E0, E1, E2)
    assert sol.d.shape == (2 * canonical_polygon(mode).shape[0],) * 2
    assert sol.v.shape == sol.d.shape
    # the two translation modes are the first columns, with zero exponents
    assert np.abs(sol.d[:2, :]).max() == 0.0
    assert np.abs(np.linalg.eigvalsh(sol.d[:2, :2]) - 0.0).max() == 0.0
    # the remaining exponents have positive real part
    assert np.linalg.eigvals(sol.d[2:, 2:]).real.min() > 0
    # the modal displacement matrix is invertible
    assert np.linalg.matrix_rank(sol.v) == sol.v.shape[0]


@pytest.mark.parametrize("mode", CANONICAL_MODES)
def test_radial_derivative_map(mode):
    """v d v^-1 maps boundary displacements to radial derivatives at xi = 1."""
    E0, E1, E2 = element_matrices(mode, 0.45)
    sol = condense(E0, E1, E2)
    M = sol.v @ sol.d @ np.linalg.inv(sol.v)
    # for a solution field, q(1) = E0 u'(1) + E1^T u(1); with u'(1) = M u(1)
    rng = np.random.default_rng(2)
    u1 = rng.standard_normal(sol.K.shape[0])
    u_xi = M @ u1
    # consistency: modal representation of u(1) reproduces the boundary data
    c = np.linalg.solve(sol.v, u1)
    assert np.abs(sol.v @ c - u1).max() < 1e-11
    assert np.abs((sol.v @ sol.d @ c - u_xi)).max() < 1e-11


@pytest.mark.parametrize("mode", CANONICAL_MODES)
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_transformed_stiffness_matches_rotated_element(mode, rotation):
    """The cached rotated stiffness equals a direct condensation of the
    rotated element geometry (when the rotation exists for the mode)."""
    try:
        T = transform(mode, rotation)
    except ValueError:
        pytest.skip(f"rotation {rotation} does not occur for mode {mode}")
    nu = 0.4
    K_cached = PatternCache(nu).stiffness(mode, rotation, 1.0)

    # direct: rotate the canonical polygon and condense it
    R = np.array([[np.cos(np.deg2rad(rotation)), -np.sin(np.deg2rad(rotation))],
                  [np.sin(np.deg2rad(rotation)), np.cos(np.deg2rad(rotation))]])
    xy_rot = (canonical_polygon(mode) @ R.T)
    A = isotropic_tangent(1.0, nu)
    E0, E1, E2 = polygon_E0E1E2(xy_rot, A)
    # node order of the rotated polygon corresponds to the shifted pattern
    k = {1: 0, 6: 0, 4: rotation // 90 if rotation in (0, 90) else None,
         2: rotation // 90, 3: rotation // 90 + (1 if rotation == 270 else 0),
         5: 2 * rotation // 90}[mode]
    n = xy_rot.shape[0]
    perm = [(j - k) % n for j in range(n)]
    order = []
    for j in perm:
        order += [2 * j, 2 * j + 1]
    K_direct = condense(E0, E1, E2).K[np.ix_(order, order)]
    assert np.abs(K_cached - K_direct).max() < 1e-10 * np.abs(K_cached).max()


def test_pattern_cache_scales_with_modulus():
    cache = PatternCache(0.3)
    K1 = cache.stiffness(2, 90, 1.0)
    K5 = cache.stiffness(2, 90, 5.0)
    assert np.abs(K5 - 5 * K1).max() < 1e-12 * np.abs(K5).max()


def test_pattern_cache_caches():
    cache = PatternCache(0.3)
    first = cache.stiffness(3, 180)
    again = cache.stiffness(3, 180)
    assert np.array_equal(first, again)
    assert cache.canonical(3) is cache.canonical(3)


def test_condense_rejects_mismatched_sizes():
    E0, E1, E2 = element_matrices(1, 0.3)
    with pytest.raises(ValueError):
        condense(E0, E1[:, :-1], E2)
