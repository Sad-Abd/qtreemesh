"""
SBFEM coefficient matrices and canonical quadtree geometry.

An S-element is a polygon with a scaling centre; its boundary is discretized
into straight line elements with two nodes each. The scaled boundary
displacement field is u(xi, eta) = N(eta) u(xi) with the radial coordinate
xi in [0, 1] and the circumferential coordinate eta in [-1, 1] per edge.
The coefficient matrices of the SBFEM equilibrium equation

    E0 xi^2 u'' + (E0 + E1^T - E1) xi u' - E2 u = 0

are boundary integrals

    E0 = int B1^T A B1 Jb deta,   E1 = int B2^T A B1 Jb deta,
    E2 = int B2^T A B2 Jb deta,

where A is the material tangent on the displacement-gradient components
(F11, F22, F12, F21) and B1 = b1(eta) Nu(eta), B2 = b2(eta) Nu'(eta) are the
circumferential and radial operators of each edge. For straight edges with
linear shape functions these operators are linear in eta and the edge
Jacobian Jb is constant, so every integrand is quadratic and the integrals
are evaluated exactly in closed form.
"""

import numpy as np

__all__ = [
    "CANONICAL_MODES",
    "canonical_polygon",
    "edge_operators",
    "isotropic_tangent",
    "polygon_E0E1E2",
    "rotation_matrix",
    "transform",
]

CANONICAL_MODES = (1, 2, 3, 4, 5, 6)

# Corners of the reference square, counter-clockwise from the bottom-left.
# Edge i runs from corner i to corner (i + 1) % 4: 0 = bottom, 1 = right,
# 2 = top, 3 = left.
_CORNERS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

# Edges carrying a hanging node for each basic quadtree cell pattern
# (the rotation = 0 instance).
_HANGING_EDGES = {
    1: [],
    2: [0],
    3: [0, 1],
    4: [0, 2],
    5: [1, 2, 3],
    6: [0, 1, 2, 3],
}


def isotropic_tangent(E, nu, formulation="plane_strain"):
    """
    Isotropic linear-elastic tangent on the displacement-gradient components
    (F11, F22, F12, F21).

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    formulation : str, optional
        "plane_strain" (default) or "plane_stress".

    Returns
    -------
    numpy array
        4x4 symmetric tangent A with sigma = A @ [eps_xx, eps_yy, eps_xy, eps_xy].
    """
    if formulation == "plane_strain":
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    elif formulation == "plane_stress":
        lmbda = E * nu / (1.0 - nu**2)
    else:
        raise ValueError(f"unknown formulation {formulation!r}")
    mu = E / (2.0 * (1.0 + nu))
    return np.array(
        [
            [lmbda + 2.0 * mu, lmbda, 0.0, 0.0],
            [lmbda, lmbda + 2.0 * mu, 0.0, 0.0],
            [0.0, 0.0, mu, mu],
            [0.0, 0.0, mu, mu],
        ]
    )


def _edge_matrices(p, q):
    """
    Closed-form integration data of one straight edge from p to q.

    Both points are relative to the scaling centre. Returns the constant edge
    Jacobian Jb and the linear operators b1 (constant) and b2 = b2_0 + eta b2_1
    in the factorised form B1 = b1 Nu, B2 = b2 Nu'.
    """
    d = q - p
    m = 0.5 * (p + q)
    Jb = 0.5 * (m[0] * d[1] - m[1] * d[0])
    if Jb == 0.0:
        raise ValueError("degenerate edge: scaling centre lies on the edge line")
    b1 = (1.0 / Jb) * np.array(
        [[d[1] / 2, 0.0], [0.0, -d[0] / 2], [-d[0] / 2, 0.0], [0.0, d[1] / 2]]
    )
    b2_0 = (1.0 / Jb) * np.array(
        [[-m[1], 0.0], [0.0, m[0]], [m[0], 0.0], [0.0, -m[1]]]
    )
    b2_1 = (1.0 / Jb) * np.array(
        [[-d[1] / 2, 0.0], [0.0, d[0] / 2], [d[0] / 2, 0.0], [0.0, -d[1] / 2]]
    )
    return Jb, b1, b2_0, b2_1


def edge_operators(p, q, eta):
    """
    Circumferential and radial operators of one straight edge.

    Parameters
    ----------
    p, q : numpy array
        Edge start and end points (2,), relative to the scaling centre.
    eta : float
        Circumferential coordinate within the edge, in [-1, 1].

    Returns
    -------
    B1, B2 : numpy array
        (4, 4) operators acting on the edge's local displacement vector
        [ux_1, uy_1, ux_2, uy_2].
    """
    Jb, b1, b2_0, b2_1 = _edge_matrices(p, q)
    b2 = b2_0 + eta * b2_1
    B1 = np.hstack([b1 * ((1.0 - eta) / 2.0), b1 * ((1.0 + eta) / 2.0)])
    B2 = np.hstack([-b2 / 2.0, b2 / 2.0])
    return B1, B2


def polygon_E0E1E2(xy, A):
    """
    Exact SBFEM coefficient matrices of an S-element.

    Parameters
    ----------
    xy : numpy array
        Polygon boundary nodes (n, 2), counter-clockwise, relative to the
        scaling centre. Edges are straight with two nodes each.
    A : numpy array
        4x4 material tangent (see `isotropic_tangent`).

    Returns
    -------
    E0, E1, E2 : numpy array
        (2n, 2n) coefficient matrices.
    """
    xy = np.asarray(xy, dtype=float)
    n = xy.shape[0]
    E0 = np.zeros((2 * n, 2 * n))
    E1 = np.zeros((2 * n, 2 * n))
    E2 = np.zeros((2 * n, 2 * n))
    for i in range(n):
        p, q = xy[i], xy[(i + 1) % n]
        Jb, b1, b2_0, b2_1 = _edge_matrices(p, q)
        # B1 = [b1 N1 | b1 N2], B2 = [-b2/2 | b2/2] with b2 = b2_0 + eta b2_1;
        # every integrand is quadratic in eta, so with
        # int_{-1}^{1} 1 deta = 2 and int eta^2 deta = 2/3:
        B10 = np.hstack([b1 / 2.0, b1 / 2.0])
        B11 = np.hstack([-b1 / 2.0, b1 / 2.0])
        B20 = np.hstack([-b2_0 / 2.0, b2_0 / 2.0])
        B21 = np.hstack([-b2_1 / 2.0, b2_1 / 2.0])
        dm = [2 * i, 2 * i + 1, 2 * ((i + 1) % n), 2 * ((i + 1) % n) + 1]
        E0[np.ix_(dm, dm)] += Jb * (
            2.0 * B10.T @ A @ B10 + (2.0 / 3.0) * B11.T @ A @ B11
        )
        E1[np.ix_(dm, dm)] += Jb * (
            2.0 * B20.T @ A @ B10 + (2.0 / 3.0) * B21.T @ A @ B11
        )
        E2[np.ix_(dm, dm)] += Jb * (
            2.0 * B20.T @ A @ B20 + (2.0 / 3.0) * B21.T @ A @ B21
        )
    return E0, E1, E2


def canonical_polygon(mode):
    """
    Local geometry of a basic quadtree cell pattern.

    The polygon is built from the reference square by inserting hanging nodes
    at the midpoints of the pattern's hanging edges; the result is centred at
    the mean of its nodes, which is the scaling-centre convention of
    quadtree-generated S-elements.

    Parameters
    ----------
    mode : int
        Basic pattern number 1-6 (1 = plain square, 6 = all edges hanging).

    Returns
    -------
    numpy array
        (n, 2) node coordinates, counter-clockwise, n = 4 + number of
        hanging edges.
    """
    if mode not in _HANGING_EDGES:
        raise ValueError(f"unknown mode {mode}, expected one of {CANONICAL_MODES}")
    nodes = []
    for i in range(4):
        nodes.append(_CORNERS[i])
        if i in _HANGING_EDGES[mode]:
            nodes.append(0.5 * (_CORNERS[i] + _CORNERS[(i + 1) % 4]))
    nodes = np.array(nodes)
    return nodes - nodes.mean(axis=0)


def rotation_matrix(degrees):
    """
    2D rotation matrix for a counter-clockwise angle in degrees.
    """
    th = np.deg2rad(degrees)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def _node_shift(mode, rotation):
    """
    Node-index shift k of a rotated cell pattern: node j of the basic
    (rotation = 0) pattern corresponds to node (j + k) % n of the rotated
    pattern. Mirrors the node-rolling semantics of the mesh generator's
    hanging-node treatment.
    """
    if mode in (1, 6):
        if rotation != 0:
            raise ValueError(f"mode {mode} never carries a nonzero rotation")
        return 0
    if mode == 4:
        if rotation not in (0, 90):
            raise ValueError(f"mode 4 only carries rotation 0 or 90, got {rotation}")
        return rotation // 90
    if mode in (2, 3, 5):
        if rotation not in (0, 90, 180, 270):
            raise ValueError(
                f"rotation must be a multiple of 90 in [0, 270], got {rotation}"
            )
        k = rotation // 90
        if mode == 3 and rotation == 270:
            k += 1
        if mode == 5:
            k *= 2
        return k
    raise ValueError(f"unknown mode {mode}, expected one of {CANONICAL_MODES}")


def transform(mode, rotation):
    """
    Displacement transform between the basic and a rotated cell pattern.

    The returned matrix maps a displacement vector of the basic (rotation = 0)
    pattern to the rotated pattern's node ordering: u_rotated = T u_basic.
    Each node block is rotated by `rotation` degrees and moved to the rotated
    pattern's node position, so the stiffness transforms as
    K_rotated = T K_basic T^T.

    Parameters
    ----------
    mode : int
        Basic pattern number 1-6.
    rotation : int
        Rotation angle in degrees (multiple of 90; only 0 and 90 occur for
        mode 4, and 0 for modes 1 and 6).

    Returns
    -------
    numpy array
        (2n, 2n) transform matrix.
    """
    k = _node_shift(mode, rotation)
    xy = canonical_polygon(mode)
    n = xy.shape[0]
    R = rotation_matrix(rotation)
    T = np.zeros((2 * n, 2 * n))
    for j in range(n):
        sigma = (j + k) % n
        T[2 * sigma : 2 * sigma + 2, 2 * j : 2 * j + 2] = R
    return T
