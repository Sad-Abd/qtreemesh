"""Tests for the QTreeMesh pipeline: elements, VTK export, drawing, FEM output."""

import numpy as np

from qtreemesh import QTree, QTreeMesh


def blob_mesh(size=32):
    """Mesh with local refinement and 2:1 balancing, hence hanging nodes."""
    image = np.zeros((size, size), dtype=int)
    image[size // 2 - 1 : size // 2 + 3, size // 2 - 1 : size // 2 + 3] = 255
    quad = QTree(None, image, 1)
    mesh = QTreeMesh(quad, balancing=True)
    mesh.create_elements()
    return mesh


def test_create_elements_matches_leaves():
    mesh = blob_mesh()
    assert len(mesh.elements) == len(mesh.leaves)
    for element, leaf in zip(mesh.elements, mesh.leaves):
        assert element.number == leaf.cell_number
        assert element.element_property == leaf.property
        for number, coord in zip(element.nodes_numbers, element.nodes_coordinates):
            assert np.array_equal(mesh.nodes[number - 1], coord)


def test_hanging_nodes_produce_subquad_elements():
    mesh = blob_mesh()
    assert any(len(element.nodes_numbers) != 4 for element in mesh.elements)


def test_vtk_export_structure(tmp_path):
    mesh = blob_mesh()
    path = tmp_path / "mesh.vtk"
    mesh.vtk_export(str(path))
    lines = path.read_text().splitlines()

    assert lines[0].startswith("# vtk DataFile Version 2.0")

    points_line = next(i for i, l in enumerate(lines) if l.startswith("POINTS"))
    n_points = int(lines[points_line].split()[1])
    assert n_points == mesh.nodes.shape[0]
    first_point = tuple(map(float, lines[points_line + 1].split()))
    assert first_point[:2] == tuple(mesh.nodes[0])

    cells_line = next(i for i, l in enumerate(lines) if l.startswith("CELLS"))
    n_cells = int(lines[cells_line].split()[1])
    assert n_cells == len(mesh.elements)

    cell_types_line = next(i for i, l in enumerate(lines) if l.startswith("CELL_TYPES"))
    assert all(l == "7" for l in lines[cell_types_line + 1 : cell_types_line + 1 + n_cells])

    data_line = next(i for i, l in enumerate(lines) if l.startswith("CELL_DATA"))
    assert lines[data_line + 1].startswith("SCALARS Average-Intensity")
    assert lines[data_line + 2].startswith("LOOKUP_TABLE")
    scalars = [float(l) for l in lines[data_line + 3 : data_line + 3 + n_cells]]
    assert scalars == [element.element_property for element in mesh.elements]


def test_draw_saves_figure(tmp_path):
    mesh = blob_mesh()
    filled_path = tmp_path / "filled.png"
    mesh.draw(True, "orangered", save_name=str(filled_path))
    assert filled_path.exists() and filled_path.stat().st_size > 0

    outline_path = tmp_path / "outline.png"
    mesh.draw(fill_inside=False, edge_color="black", save_name=str(outline_path))
    assert outline_path.exists() and outline_path.stat().st_size > 0


def test_adjust_mesh_for_FEM_triangulates_all_subelements():
    mesh = blob_mesh()
    nodes, fem_elements, fem_properties = mesh.adjust_mesh_for_FEM(
        force_triangulation=True
    )
    assert nodes is mesh.nodes
    assert len(fem_properties) == len(fem_elements)
    assert all(len(element) == 3 for element in fem_elements)
    assert set(fem_properties) == {element.element_property for element in mesh.elements}


def test_adjust_mesh_for_FEM_keeps_quads_without_forcing():
    mesh = blob_mesh()
    _, fem_elements, _ = mesh.adjust_mesh_for_FEM(force_triangulation=False)
    assert any(len(element) == 3 for element in fem_elements)
    assert any(len(element) == 4 for element in fem_elements)


def test_meshtree_without_balancing_keeps_fewer_cells():
    image = np.zeros((32, 32), dtype=int)
    image[8, 8] = 255
    unbalanced = QTreeMesh(QTree(None, image, 1), balancing=False)
    balanced = QTreeMesh(QTree(None, image, 1), balancing=True)
    assert unbalanced.quad_tree.count_leaves < balanced.quad_tree.count_leaves
