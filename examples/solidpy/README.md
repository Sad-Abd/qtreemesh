# SolidSpy FEM export example

Prototype for exporting a qtreemesh mesh to
[SolidsPy](https://github.com/AppliedMechanics-EAFIT/SolidsPy), an open-source finite
element code, and visualizing the analysis results.

`test2.py`:

1. meshes `5.jpg` with qtreemesh (`QTreeMesh.adjust_mesh_for_FEM`),
2. writes the FEM input files SolidSpy reads from the working directory —
   `nodes.txt`, `eles.txt`, `mater.txt`, `loads.txt` (fixed bottom edge,
   tensile load on the top edge),
3. runs the analysis via `solidspy.solids_GUI`,
4. writes the deformed result (displacements, PK1/Cauchy/PK2 stresses) to
   `output3.vtk` with `outputwrites.create_vtk_file`.

The checked-in `.txt` and `.vtk` files are the outputs of that run;
`4.jpg` and `5.jpg` are copies of the images in `examples/` so the script
is self-contained. Requires `solidspy` and `pandas` in addition to qtreemesh.
