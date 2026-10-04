# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `QTreeMesh.from_image(image, crit, scale, max_size, balancing)`: builds the
  mesh from an image array in one step (padding, quadtree, elements).
- `QTreeMesh.trim_padding(rows, cols)`: removes elements that lie entirely in
  the padded region of a preprocessed image; `pixel_to_element()` reports 0
  for padded pixels that no element covers.
- `QTreeMesh.boundary_edges()` and `QTreeMesh.boundary_nodes()`: the mesh
  boundary as (node_1, node_2) pairs and as node numbers;
  `SBFEMModel.boundary_edges` now delegates to the mesh.
- `qtreemesh.sbfem` subpackage: linear static scaled boundary finite element
  analysis of quadtree meshes. `SBFEMModel` assembles the sparse stiffness from
  the condensed solutions of the six basic cell patterns, applies Dirichlet and
  traction boundary conditions, solves, and recovers boundary stresses and
  interior fields. Requires `scipy`.
- `scipy` added to the package dependencies.

## [0.2.0] - 2026-10-04
### Changed
- Rewrote `QTreeMesh.labeling()` to assign node numbers with a hash map instead of a
  linear search over all nodes found so far. The numbering is unchanged (1-based,
  first-seen order); mesh building on large images is orders of magnitude faster.

### Added
- `QTreeMesh.pixel_to_element()`: maps every pixel of the preprocessed image to the
  1-based number of the element covering it.
- `QTreeMesh.element_labels()`: exact per-cell label for every element (the single
  intensity shared by all pixels under the cell), unambiguous for label and
  multi-material images where the averaged `element_property` is not.
- `QTree` accepts a `max_size` option (pixels): cells larger than this size are
  split even when their pixels are homogeneous, so uniform regions are no longer
  left as one huge cell. Replaces the checkerboard-image workaround.
- Test suite covering the whole package (`tests/`): node numbering, `max_size`,
  neighbor search, 2:1 balancing, image preprocessing, element mode detection and
  hanging-node treatment, VTK export, drawing, and FEM output — 100% line coverage.

## [0.1.3]
- Added the method `adjust_mesh_for_FEM` to generate FEM-compatible mesh from the QuadTreeMesh

## [0.1.2]
- Added a method 'vtk_export()' to create an unstructured grid vtk file from mesh.
- Resolved a bug related to adding midpoints to edges. 

## [0.1.1] - 2023-7-17

### Added
- Start using a Changelog.
- Added docstrings for all classes, methods and functions.

### Changed
- Modified 'draw()' method to remove axes from the figure and make it square.

### Removed
- Unnecessary attributes of QTree class ('element_numbers','element_nodes' and 'element_type').