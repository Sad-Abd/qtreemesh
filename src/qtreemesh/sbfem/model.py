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
    _node_shift,
    canonical_polygon,
    edge_operators,
    isotropic_tangent,
    rotation_matrix,
    transform,
)

__all__ = ["SBFEMModel"]


def _check_modulus(value):
    """Raise ValueError unless value is a positive, finite Young's modulus."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Young's modulus must be positive and finite, got {value}")


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

    Raises
    ------
    ValueError
        If the mesh has no elements, a modulus is not positive and finite,
        or nu or the formulation is invalid (see `isotropic_tangent`).
    KeyError
        If a cell label of the mesh has no entry in `moduli`.
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
            missing = sorted(set(labels.tolist()) - set(moduli))
            if missing:
                raise KeyError(f"no modulus given for cell label(s) {missing}")
            for value in moduli.values():
                _check_modulus(value)
        else:
            _check_modulus(moduli)
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
                    "shift": _node_shift(mode, rotation),
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
        edges = [(int(a), int(b)) for a, b in edges]
        if tractions.shape != (len(edges), 2):
            raise ValueError(
                f"tractions must have shape ({len(edges)}, 2), got {tractions.shape}"
            )
        nnodes = self.ndof // 2
        if any(not (1 <= n <= nnodes) for edge in edges for n in edge):
            raise ValueError(f"edge node numbers must lie in 1..{nnodes}")
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
            (m, 3) rows of [node, direction, value]; node is a 1-based mesh
            node number, direction 1 = x or 2 = y. May be empty.
        force : numpy array, optional
            (ndof,) external nodal force vector.

        Returns
        -------
        u, reactions : numpy array
            (ndof,) nodal displacements and (ndof,) support reaction forces
            (the residual K u - force on the constrained degrees of freedom).

        Raises
        ------
        ValueError
            If a constraint row or the force vector is malformed, or if the
            constraints leave a rigid-body motion free while the load is not
            zero (the system is then singular).
        """
        K = self.stiffness()
        nnodes = self.ndof // 2
        dirichlet = np.asarray(dirichlet, dtype=float).reshape(-1, 3)
        nodes, directions, values = dirichlet.T
        if not (np.all(nodes == np.round(nodes)) and np.all((nodes >= 1) & (nodes <= nnodes))):
            raise ValueError(f"constrained node numbers must be integers in 1..{nnodes}")
        if not np.all((directions == 1) | (directions == 2)):
            raise ValueError("constraint directions must be 1 (x) or 2 (y)")
        if not np.all(np.isfinite(values)):
            raise ValueError("constraint values must be finite")
        if force is None:
            force = np.zeros(self.ndof)
        force = np.asarray(force, dtype=float)
        if force.shape != (self.ndof,) or not np.all(np.isfinite(force)):
            raise ValueError(f"force must be a finite vector of length {self.ndof}")

        u = np.zeros(self.ndof)
        constrained = ((nodes.astype(int) - 1) * 2 + (directions.astype(int) - 1)).astype(int)
        u[constrained] = values
        free = np.setdiff1d(np.arange(self.ndof), constrained)
        f_eff = force - K[:, constrained] @ u[constrained]
        rhs = f_eff[free]
        if free.size and not self._rigid_motions_constrained(nodes, directions):
            # the rigid-body motions are null vectors of the free block, so a
            # nonzero load has no solution; a zero load is solved by zero
            if np.any(rhs != 0.0):
                raise ValueError(
                    "the constraints leave a rigid-body motion free; "
                    "constrain x and y translations and the rotation"
                )
        elif free.size:
            u[free] = spla.spsolve(K[free][:, free], rhs)
            if not np.all(np.isfinite(u)):
                raise ValueError("the stiffness matrix of the free degrees of freedom is singular")
        return u, K @ u - force

    def _rigid_motions_constrained(self, nodes, directions):
        """
        Whether the constraints remove both rigid translations and the rotation.

        The translations are fixed when some x and some y degree of freedom is
        constrained. A rotation about a centre (x0, y0) is fixed unless every
        constrained x degree of freedom lies on the line y = y0 and every
        constrained y degree of freedom on the line x = x0.
        """
        coordinates = self.mesh.nodes[nodes.astype(int) - 1]
        x_fixed = directions == 1
        y_fixed = directions == 2
        if not (x_fixed.any() and y_fixed.any()):
            return False
        rows = np.unique(coordinates[x_fixed, 1]).size
        columns = np.unique(coordinates[y_fixed, 0]).size
        return rows > 1 or columns > 1

    def _check_displacements(self, u):
        """The displacement vector as a float array of length ndof."""
        u = np.asarray(u, dtype=float)
        if u.shape != (self.ndof,):
            raise ValueError(f"u must have shape ({self.ndof},), got {u.shape}")
        return u

    def _pullback(self, element, u):
        """Canonical-frame nodal displacements of one element."""
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
        radial solution, scaled by the element's Young's modulus and rotated
        to the element's own frame. The rows follow the elements in mesh
        order and, within an element, the edges of its node list (edge i runs
        from node i to node (i + 1) % n).

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
        u = self._check_displacements(u)
        points = []
        stress = []
        for element in self._elements:
            u_c, u_xi_c, _sol = self._canonical_solution(element, u)
            xy_c = canonical_polygon(element["mode"])
            n = xy_c.shape[0]
            R = rotation_matrix(element["rotation"])
            for edge in range(n):
                # edge `edge` of the element's node list is edge
                # (edge - shift) % n of the basic pattern
                i = (edge - element["shift"]) % n
                j = (i + 1) % n
                B1, B2 = edge_operators(xy_c[i], xy_c[j], 0.0)
                dm = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
                eps_c = B1 @ u_xi_c[dm] + B2 @ u_c[dm]
                sigma_c = element["E"] * (self._A @ eps_c)
                points.append(
                    element["centre"]
                    + element["R_size"] * R @ (0.5 * (xy_c[i] + xy_c[j]))
                )
                stress.append(self._stress_rotate(element, sigma_c[:3]))
        return np.array(points), np.array(stress)

    def field(self, element_number, edge, eta, xi, u):
        """
        Displacement and stress at an interior point of one element.

        The stress includes the element's Young's modulus.

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
        if not -1.0 <= eta <= 1.0:
            raise ValueError("eta must lie in [-1, 1]")
        if int(element_number) != element_number or not 1 <= element_number <= len(self._elements):
            raise ValueError(f"element_number must be an integer in 1..{len(self._elements)}")
        u = self._check_displacements(u)
        element = self._elements[int(element_number) - 1]
        mode = element["mode"]
        xy_c = canonical_polygon(mode)
        n = xy_c.shape[0]
        if int(edge) != edge or not 0 <= edge < n:
            raise ValueError(f"edge must be an integer in 0..{n - 1}")
        edge = int(edge)

        # edge `edge` of the element's node list is edge (edge - shift) % n
        # of the basic pattern
        i = (edge - element["shift"]) % n
        j = (i + 1) % n

        sol = self._cache.canonical(mode)
        u_c = self._pullback(element, u)
        c = np.linalg.solve(sol.v, u_c)
        # radial solution u(xi) = v xi^d c; the translation modes have
        # exponent 0, the others xi^S = expm(S ln xi)
        S_pp = sol.d[2:, 2:]
        xi_S = sla.expm(S_pp * np.log(xi))
        grow = np.zeros_like(sol.d)
        grow[:2, :2] = np.eye(2)
        grow[2:, 2:] = xi_S
        rate = np.zeros_like(sol.d)
        rate[2:, 2:] = S_pp @ xi_S / xi
        u_c_xi_nodal = sol.v @ grow @ c
        du_c_xi_nodal = sol.v @ rate @ c

        p_c, q_c = xy_c[i], xy_c[j]
        B1, B2 = edge_operators(p_c, q_c, eta)
        N1 = (1.0 - eta) / 2.0
        N2 = (1.0 + eta) / 2.0
        dm = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        # strain = B1 u'(xi) + B2 u(xi) / xi
        eps_c = B1 @ du_c_xi_nodal[dm] + B2 @ u_c_xi_nodal[dm] / xi
        sigma_c = element["E"] * (self._A @ eps_c)

        R = rotation_matrix(element["rotation"])
        disp_c = np.array(
            [
                N1 * u_c_xi_nodal[dm[0]] + N2 * u_c_xi_nodal[dm[2]],
                N1 * u_c_xi_nodal[dm[1]] + N2 * u_c_xi_nodal[dm[3]],
            ]
        )
        x_c = xi * (N1 * p_c + N2 * q_c)
        point = element["centre"] + element["R_size"] * R @ x_c
        displacement = element["R_size"] * R @ disp_c
        return displacement, self._stress_rotate(element, sigma_c[:3]), point
