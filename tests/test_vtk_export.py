"""Tests for the vtk export options."""

import numpy as np

from qtreemesh import QTreeMesh


def disk_mesh(size=32):
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size), dtype=int)
    image[((xx - size // 2) ** 2 + (yy - size // 2) ** 2) < size**2 // 16] = 3
    return QTreeMesh.from_image(image, crit=1)


def read_sections(path):
    lines = path.read_text().splitlines()

    def section_start(prefix):
        return next(i for i, l in enumerate(lines) if l.startswith(prefix))

    i_cells = section_start("CELLS")
    n_cells = int(lines[i_cells].split()[1])
    i_types = section_start("CELL_TYPES")
    types = [int(l) for l in lines[i_types + 1 : i_types + 1 + n_cells]]
    sizes = [int(lines[i_cells + 1 + k].split()[0]) for k in range(n_cells)]

    scalars = {}
    i_data = section_start("CELL_DATA")
    k = i_data + 1
    while k < len(lines):
        if lines[k].startswith("SCALARS "):
            name = lines[k].split()[1]
            values = [
                float(l) for l in lines[k + 2 : k + 2 + n_cells]
            ]
            scalars[name] = values
            k += 2 + n_cells
        else:
            k += 1
    return types, sizes, scalars


def test_default_export_keeps_polygon_cells(tmp_path):
    mesh = disk_mesh()
    path = tmp_path / "mesh.vtk"
    mesh.vtk_export(str(path))
    types, sizes, scalars = read_sections(path)
    assert set(types) == {7}
    assert len(types) == len(mesh.elements)
    assert set(scalars) == {"Average-Intensity"}
    assert scalars["Average-Intensity"] == [e.element_property for e in mesh.elements]


def test_adjusted_export_triangle_cells(tmp_path):
    mesh = disk_mesh()
    path = tmp_path / "mesh.vtk"
    mesh.vtk_export(str(path), adjusted=True)
    types, sizes, scalars = read_sections(path)
    assert set(types) == {5}
    assert set(sizes) == {3}
    expected = sum(len(e.quad_treatment(True)) for e in mesh.elements)
    assert len(types) == expected
    assert scalars["Average-Intensity"] == [
        e.element_property for e in mesh.elements for _ in e.quad_treatment(True)
    ]


def test_adjusted_export_mixed_cells(tmp_path):
    mesh = disk_mesh()
    path = tmp_path / "mesh.vtk"
    mesh.vtk_export(str(path), adjusted=True, force_triangulation=False)
    types, sizes, _ = read_sections(path)
    assert set(types) == {5, 9}
    assert set(sizes) <= {3, 4}


def test_cell_data_arrays(tmp_path):
    mesh = disk_mesh()
    path = tmp_path / "mesh.vtk"
    mesh.vtk_export(str(path), cell_data=True)
    types, _, scalars = read_sections(path)
    assert set(scalars) == {"Label", "Mode", "Rotation", "Size", "Average-Intensity"}
    labels = mesh.element_labels(strict=False)
    assert scalars["Label"] == [float(v) for v in labels]
    assert scalars["Mode"] == [float(e.element_type[0]) for e in mesh.elements]
    assert scalars["Rotation"] == [float(e.element_type[1]) for e in mesh.elements]
    assert scalars["Size"] == [float(e.element_type[2]) for e in mesh.elements]


def test_adjusted_cell_data_expanded_per_sub_cell(tmp_path):
    mesh = disk_mesh()
    path = tmp_path / "mesh.vtk"
    mesh.vtk_export(str(path), adjusted=True, cell_data=True)
    types, _, scalars = read_sections(path)
    labels = mesh.element_labels(strict=False)
    expected_label = [
        float(label)
        for element, label in zip(mesh.elements, labels)
        for _ in element.quad_treatment(True)
    ]
    assert scalars["Label"] == expected_label
    expected_mode = [
        float(element.element_type[0])
        for element in mesh.elements
        for _ in element.quad_treatment(True)
    ]
    assert scalars["Mode"] == expected_mode
