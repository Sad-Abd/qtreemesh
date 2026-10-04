"""
Linear static SBFEM analysis of a quadtree mesh.

`SBFEMModel` turns a `QTreeMesh` into a sparse linear system and solves it:
every mesh cell is an S-element whose condensed stiffness comes from the
pattern cache (one basic pattern per cell shape, rotated and scaled by the
cell's modulus), boundary conditions are applied on the mesh nodes, and the
solution is recovered as boundary stresses and interior fields.

Displacements are collected in a single vector ordered
[ux_0, uy_0, ux_1, uy_1, ...] over the mesh nodes (1-based numbering as in
the mesh). A Dirichlet constraint is a row [node, direction, value] with
direction 1 = x and 2 = y.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla

from .condense import PatternCache
from .patterns import (
    canonical_polygon,
    edge_operators,
    isotropic_tangent,
    rotation_matrix,
    transform,
)

__all__ = ["SBFEMModel"]


class SBFEMModel:
    """
    Linear static SBFEM model of a quadtree mesh.

    Parameters
    ----------
    mesh : QTreeMesh
        Mesh with elements already generated (`create_elements` called).
        Each element must carry its pattern `[mode, rotation, size]` in
        `element_type` and its counter-clockwise node list in
        `nodes_numbers`.
    moduli : float or dict
        Young's modulus of the material. A float applies to the whole model;
        a dict maps the exact per-cell label (see `QTreeMesh.element_labels`)
        to the modulus of that region and requires a label-homogeneous mesh.
    nu : float
        Poisson's ratio.
    formulation : str, optional
        "plane_strain" (default) or "plane_stress".

    Attributes
    ----------
    ndof : int
        Number of degrees of freedom (twice the number of mesh nodes).
    """

    def __init__(self, mesh, moduli, nu, formulation="plane_strain"):
        if not mesh.elements:
            raise ValueError("mesh has no elements; call create_elements first")
        self.mesh = mesh
        self.nu = nu
        self.formulation = formulation
        self.ndof = 2 * mesh.nodes.shape[0]
        self._cache = PatternCache(nu, formulation)
        self._A = isotropic_tangent(1.0, nu, formulation)
        self._K = None

        if isinstance(moduli, dict):
            labels = mesh.element_labels()
        self._elements = []
        for number, element in enumerate(mesh.elements, start=1):
            mode, rotation, _size = element.element_type
            nodes = np.asarray(element.nodes_numbers, dtype=int)
            xy = np.asarray(element.nodes_coordinates, dtype=float)
            extent = xy[:, 0].max() - xy[:, 0].min()
            if isinstance(moduli, dict):
                E = moduli[labels[number - 1]]
            else:
                E = moduli
            self._elements.append(
                {
                    "number": number,
                    "nodes": nodes,
                    "dofs": np.stack([2 * (nodes - 1), 2 * (nodes - 1) + 1]).ravel(order="F"),
                    "mode": mode,
                    "rotation": rotation,
                    "E": float(E),
                    "centre": xy.mean(axis=0),
                    "R_size": extent / 2.0,
                    "xy": xy,
                }
            )

    def stiffness(self):
        """
        Sparse global stiffness matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            (ndof, ndof) stiffness assembled from the condensed element
            stiffnesses.
        """
        if self._K is None:
            rows, cols, vals = [], [], []
            for element in self._elements:
                K = self._cache.stiffness(element["mode"], element["rotation"], element["E"])
                dofs = element["dofs"]
                rows.append(np.repeat(dofs, len(dofs)))
                cols.append(np.tile(dofs, len(dofs)))
                vals.append(K.ravel())
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            vals = np.concatenate(vals)
            self._K = sp.csr_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof))
        return self._K

    def boundary_edges(self):
        """
        Edges of the mesh that belong to a single element.

        Returns
        -------
        list of tuple
            (node_1, node_2) pairs of 1-based node numbers.
        """
        counts = {}
        for element in self._elements:
            nodes = element["nodes"]
            n = len(nodes)
            for i in range(n):
                a, b = int(nodes[i]), int(nodes[(i + 1) % n])
                key = (a, b) if a < b else (b, a)
                counts[key] = counts.get(key, 0) + 1
        return [edge for edge, count in counts.items() if count == 1]

    def traction_forces(self, edges, tractions):
        """
        Consistent nodal forces of constant tractions on boundary edges.

        Parameters
        ----------
        edges : sequence of tuple
            (node_1, node_2) pairs of 1-based node numbers.
        tractions : numpy array
            (m, 2) traction vectors [tx, ty] per edge (force per unit length).

        Returns
        -------
        numpy array
            (ndof,) force vector.
        """
        tractions = np.asarray(tractions, dtype=float)
        force = np.zeros(self.ndof)
        for (n1, n2), t in zip(edges, tractions):
            length = np.linalg.norm(self.mesh.nodes[n2 - 1] - self.mesh.nodes[n1 - 1])
            force[2 * (n1 - 1) : 2 * (n1 - 1) + 2] += 0.5 * length * t
            force[2 * (n2 - 1) : 2 * (n2 - 1) + 2] += 0.5 * length * t
        return force

    def solve(self, dirichlet, force=None):
        """
        Solve the static boundary-value problem.

        Parameters
        ----------
        dirichlet : numpy array
            (m, 3) rows of [node, direction, value]; direction 1 = x,
            2 = y. May be empty.
        force : numpy array, optional
            (ndof,) external nodal force vector.

        Returns
        -------
        u, reactions : numpy array
            (ndof,) nodal displacements and (ndof,) support reaction forces
            (the residual K u - force on the constrained degrees of freedom).
        """
        K = self.stiffness()
        u = np.zeros(self.ndof)
        dirichlet = np.asarray(dirichlet, dtype=float).reshape(-1, 3)
        if force is None:
            force = np.zeros(self.ndof)
        force = np.asarray(force, dtype=float)
        if dirichlet.shape[0]:
            constrained = ((dirichlet[:, 0].astype(int) - 1) * 2 + (dirichlet[:, 1].astype(int) - 1)).astype(int)
            u[constrained] = dirichlet[:, 2]
        else:
            constrained = np.array([], dtype=int)
        free = np.setdiff1d(np.arange(self.ndof), constrained)
        f_eff = force - K[:, constrained] @ u[constrained]
        u[free] = spla.spsolve(K[free][:, free], f_eff[free])
        return u, K @ u - force

    def _pullback(self, element, u):
        """Canonical-frame nodal displacements of one element."""
        sol = self._cache.canonical(element["mode"])
        T = transform(element["mode"], element["rotation"])
        return T.T @ u[element["dofs"]] / element["R_size"]

    def _stress_rotate(self, element, sigma_c):
        """Rotate canonical stress components (sxx, syy, sxy) to the element frame."""
        R = rotation_matrix(element["rotation"])
        sxx, syy, sxy = sigma_c
        return np.array(
            [
                R[0, 0] ** 2 * sxx + 2 * R[0, 0] * R[0, 1] * sxy + R[0, 1] ** 2 * syy,
                R[1, 0] ** 2 * sxx + 2 * R[1, 0] * R[1, 1] * sxy + R[1, 1] ** 2 * syy,
                (R[0, 0] * R[1, 0] * sxx + (R[0, 0] * R[1, 1] + R[0, 1] * R[1, 0]) * sxy
                 + R[0, 1] * R[1, 1] * syy),
            ]
        )

    def _canonical_solution(self, element, u):
        """(u_c, u_xi_c, sol) of one element: boundary and radial-derivative
        nodal displacement vectors in the canonical frame."""
        sol = self._cache.canonical(element["mode"])
        u_c = self._pullback(element, u)
        u_xi_c = sol.v @ (sol.d @ np.linalg.solve(sol.v, u_c))
        return u_c, u_xi_c, sol

    def boundary_stress(self, u):
        """
        Stress at the midpoint of every element edge.

        The stress is evaluated in the basic pattern's frame from the element's
        radial solution and rotated to the element's own frame.

        Parameters
        ----------
        u : numpy array
            (ndof,) nodal displacement vector, e.g. from `solve`.

        Returns
        -------
        points, stress : numpy array
            (m, 2) midpoint coordinates and (m, 3) stress components
            [sigma_xx, sigma_yy, sigma_xy], one row per element edge.
        """
        points = []
        stress = []
        for element in self._elements:
            u_c, u_xi_c, _sol = self._canonical_solution(element, u)
            xy_c = canonical_polygon(element["mode"])
            n = xy_c.shape[0]
            for i in range(n):
                j = (i + 1) % n
                B1, B2 = edge_operators(xy_c[i], xy_c[j], 0.0)
                dm = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
                eps_c = B1 @ u_xi_c[dm] + B2 @ u_c[dm]
                sigma_c = self._A @ eps_c
                R = rotation_matrix(element["rotation"])
                points.append(
                    element["centre"]
                    + element["R_size"] * R @ (0.5 * (xy_c[i] + xy_c[j]))
                )
                stress.append(self._stress_rotate(element, sigma_c[:3]))
        return np.array(points), np.array(stress)

    def field(self, element_number, edge, eta, xi, u):
        """
        Displacement and stress at an interior point of one element.

        The point is given in the S-element's polar coordinates: `edge` and
        `eta` locate the point on the element boundary (edge index within the
        element's node list, circumferential coordinate in [-1, 1]) and `xi`
        is the radial coordinate, x = centre + xi * (boundary point - centre).

        Parameters
        ----------
        element_number : int
            1-based element number (as in `QTreeElement.number`).
        edge : int
            Edge index 0..n-1 within the element's node list (edge i runs
            from node i to node (i+1) % n).
        eta : float
            Circumferential coordinate in [-1, 1].
        xi : float
            Radial coordinate in (0, 1].
        u : numpy array
            (ndof,) nodal displacement vector.

        Returns
        -------
        displacement, stress, point : numpy array
            (2,) displacement vector, (3,) stress components
            [sigma_xx, sigma_yy, sigma_xy] and (2,) coordinates of the point.
        """
        if not 0.0 < xi <= 1.0:
            raise ValueError("xi must lie in (0, 1]")
        element = self._elements[element_number - 1]
        mode = element["mode"]
        xy_c = canonical_polygon(mode)
        n = xy_c.shape[0]
        if not 0 <= edge < n:
            raise ValueError(f"edge must be in 0..{n - 1}")

        sol = self._cache.canonical(mode)
        u_c = self._pullback(element, u)
        c = np.linalg.solve(sol.v, u_c)
        S_pp = sol.d[2:, 2:]
        xi_S = sla.fractional_matrix_power(S_pp, xi)
        grow = np.zeros_like(sol.d)
        grow[:2, :2] = np.eye(2)
        grow[2:, 2:] = xi_S
        rate = np.zeros_like(sol.d)
        rate[2:, 2:] = S_pp @ xi_S / xi
        u_c_xi_nodal = sol.v @ grow @ c
        du_c_xi_nodal = sol.v @ rate @ c

        j = (edge + 1) % n
        p_c, q_c = xy_c[edge], xy_c[j]
        B1, B2 = edge_operators(p_c, q_c, eta)
        N1 = (1.0 - eta) / 2.0
        N2 = (1.0 + eta) / 2.0
        dm = [2 * edge, 2 * edge + 1, 2 * j, 2 * j + 1]
        eps_c = B1 @ du_c_xi_nodal[dm] + B2 @ u_c_xi_nodal[dm]
        sigma_c = self._A @ eps_c

        R = rotation_matrix(element["rotation"])
        u_local = np.array([u_c_xi_nodal[2 * edge], u_c_xi_nodal[2 * edge + 1],
                            u_c_xi_nodal[2 * j], u_c_xi_nodal[2 * j + 1]])
        disp_c = np.array(
            [N1 * u_local[0] + N2 * u_local[2], N1 * u_local[1] + N2 * u_local[3]]
        )
        x_c = xi * (N1 * p_c + N2 * q_c)
        point = element["centre"] + element["R_size"] * R @ x_c
        displacement = element["R_size"] * R @ disp_c
        return displacement, self._stress_rotate(element, sigma_c[:3]), point
