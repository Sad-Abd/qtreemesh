"""
Condensation of an S-element: from coefficient matrices to stiffness and
modal data.

The condensed stiffness and the radial solution of an S-element follow from
the Hamiltonian block matrix

    Z = [[-E0^-1 E1^T, E0^-1], [E2 - E1 E0^-1 E1^T, E1 E0^-1]]

whose eigenvalues come in +/- pairs. The two rigid translations of the
S-element are exact null vectors of E1^T and E2; they correspond to a
four-dimensional zero-eigenvalue block of Z (two 2x2 Jordan blocks) that
would make a half-plane spectrum split ill-defined. `condense` removes this
block a priori: with T the orthonormal translation basis and U an
orthonormal complement (P = [T U] applied identically to the displacement
and the force half of the phase space), only the (U-displacement, U-force)
sub-block Z_red of Z is solved. Z_red has no eigenvalues on the imaginary
axis, so an ordered real Schur decomposition Z_red = V S V^T with the
right-half-plane eigenvalues first is well defined and gives

    K = U Kr U^T,   Kr = Vq Vu^-1,

with Vu, Vq the displacement and force parts of the right-half-plane
eigenvectors. The full radial solution acts on

    u(xi) = T c_T + (U Vu + T a) xi^S c,

where the translation components a of the remaining modes follow from the
invariance of the full phase space, a S = T^T(A U Vu + G U Vq) with
A = -E0^-1 E1^T and G = E0^-1. This is returned as the modal data
d = diag(0, S) and v = [T | U Vu + T a], so that with c = v^-1 u(1) the
radial derivative at the boundary is u'(1) = v d v^-1 u(1).
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .patterns import (
    canonical_polygon,
    isotropic_tangent,
    polygon_E0E1E2,
    transform,
)

__all__ = ["SElementSolution", "condense", "PatternCache"]


@dataclass
class SElementSolution:
    """
    Condensed solution of one S-element.

    Attributes
    ----------
    K : numpy array
        (2n, 2n) condensed stiffness at the boundary xi = 1.
    d : numpy array
        (2n, 2n) modal exponent matrix, diag(0, S) with S the ordered real
        Schur block of the reduced Hamiltonian.
    v : numpy array
        (2n, 2n) modal displacement matrix; column j is the displacement
        part of solution mode j, satisfying Z [v; w] = [v; w] d for the
        force part w.
    """

    K: np.ndarray
    d: np.ndarray
    v: np.ndarray


def condense(E0, E1, E2):
    """
    Condense an S-element from its coefficient matrices.

    Parameters
    ----------
    E0, E1, E2 : numpy array
        (2n, 2n) SBFEM coefficient matrices (see `patterns.polygon_E0E1E2`).

    Returns
    -------
    SElementSolution
        Condensed stiffness K and modal data (d, v).

    Raises
    ------
    ValueError
        If the matrices are not square and of equal size.
    numpy.linalg.LinAlgError
        If the reduced Hamiltonian does not split into two half-plane
        spectra of equal size.
    """
    E0 = np.asarray(E0, dtype=float)
    E1 = np.asarray(E1, dtype=float)
    E2 = np.asarray(E2, dtype=float)
    nd = E0.shape[0]
    n = nd // 2
    if E1.shape != (nd, nd) or E2.shape != (nd, nd):
        raise ValueError("E0, E1 and E2 must be square and of equal size")

    # rigid translations: exact null vectors of E1 and E2 (orthonormal basis)
    T = np.zeros((nd, 2))
    T[0::2, 0] = 1.0
    T[1::2, 1] = 1.0
    T /= np.sqrt(n)
    # orthonormal completion U (the first two columns of Q span T)
    Q, _ = np.linalg.qr(np.hstack([T, np.eye(nd)]))
    U = Q[:, 2:]
    m = nd - 2

    E0i = np.linalg.inv(E0)
    A_z = -E0i @ E1.T
    G_z = E0i
    Q_z = E2 - E1 @ E0i @ E1.T
    Z = np.block([[A_z, G_z], [Q_z, -A_z.T]])

    # transform the phase space to (T-disp, U-disp, T-force, U-force) and keep
    # the (U-displacement, U-force) sub-block
    Qp = np.zeros((2 * nd, 2 * nd))
    Qp[:nd, :nd] = Q
    Qp[nd:, nd:] = Q
    Z = Qp.T @ Z @ Qp
    idx = np.concatenate([np.arange(2, nd), nd + 2 + np.arange(m)])
    Z_red = Z[np.ix_(idx, idx)]

    S_full, V_red, sdim = sla.schur(Z_red, output="real", sort="rhp")
    if sdim != m:
        raise np.linalg.LinAlgError(
            f"the reduced Hamiltonian has {sdim} right-half-plane eigenvalues, "
            f"expected {m}"
        )
    S_pp = S_full[:m, :m]
    W = V_red[:, :m]
    V_u = W[:m, :]
    V_q = W[m:, :]

    K_r = np.linalg.solve(V_u.T, V_q.T).T
    K = U @ K_r @ U.T
    K = 0.5 * (K + K.T)  # symmetric in exact arithmetic

    # modal data: translation modes (xi-independent) + lifted reduced modes
    a = np.linalg.solve(S_pp.T, (T.T @ (A_z @ U @ V_u + G_z @ U @ V_q)).T).T
    d = np.zeros((nd, nd))
    d[2:, 2:] = S_pp
    v = np.hstack([T, U @ V_u + T @ a])
    return SElementSolution(K=K, d=d, v=v)


class PatternCache:
    """
    Condensed solutions of the basic quadtree cell patterns at one material
    state.

    Every balanced quadtree cell is a rotated instance of one of the six
    basic patterns; the canonical (rotation = 0) solution is condensed once
    per pattern and transformed for each rotation. The canonical solution is
    built at unit Young's modulus: the stiffness of any cell is the transformed
    canonical stiffness scaled by its modulus.

    Parameters
    ----------
    nu : float
        Poisson's ratio.
    formulation : str, optional
        "plane_strain" (default) or "plane_stress".
    """

    def __init__(self, nu, formulation="plane_strain"):
        self.nu = nu
        self.formulation = formulation
        self._canonical = {}
        self._stiffness = {}

    def canonical(self, mode):
        """
        Condensed solution of the basic (rotation = 0) pattern.

        Parameters
        ----------
        mode : int
            Basic pattern number 1-6.

        Returns
        -------
        SElementSolution
            Solution at unit Young's modulus, in the canonical frame.
        """
        if mode not in self._canonical:
            xy = canonical_polygon(mode)
            A = isotropic_tangent(1.0, self.nu, self.formulation)
            E0, E1, E2 = polygon_E0E1E2(xy, A)
            self._canonical[mode] = condense(E0, E1, E2)
        return self._canonical[mode]

    def stiffness(self, mode, rotation, E=1.0):
        """
        Condensed stiffness of a cell pattern instance.

        Parameters
        ----------
        mode : int
            Basic pattern number 1-6.
        rotation : int
            Rotation angle in degrees of the instance.
        E : float, optional
            Young's modulus of the cell (default 1.0).

        Returns
        -------
        numpy array
            (2n, 2n) condensed stiffness in the instance's node ordering.
        """
        key = (mode, rotation)
        if key not in self._stiffness:
            T = transform(mode, rotation)
            self._stiffness[key] = T @ self.canonical(mode).K @ T.T
        return E * self._stiffness[key]
