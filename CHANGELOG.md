# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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