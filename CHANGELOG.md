# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- Rewrote `QTreeMesh.labeling()` to assign node numbers with a hash map instead of a
  linear search over all nodes found so far. The numbering is unchanged (1-based,
  first-seen order); mesh building on large images is orders of magnitude faster.

### Added
- `QTree` accepts a `max_size` option (pixels): cells larger than this size are
  split even when their pixels are homogeneous, so uniform regions are no longer
  left as one huge cell. Replaces the checkerboard-image workaround.
- Test suite covering the node-numbering contract (`tests/test_labeling.py`) and
  the `max_size` option (`tests/test_max_size.py`).

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