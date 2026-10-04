"""Tests for drawing the mesh."""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from qtreemesh import QTreeMesh


def disk_mesh(size=32):
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size), dtype=int)
    image[((xx - size // 2) ** 2 + (yy - size // 2) ** 2) < size**2 // 16] = 3
    return QTreeMesh.from_image(image, crit=1)


def test_draw_returns_figure_and_saves(tmp_path):
    mesh = disk_mesh()
    path = tmp_path / "mesh.png"
    fig = mesh.draw(show=False, save_name=str(path))
    assert fig is not None
    assert path.exists() and path.stat().st_size > 0


def test_draw_fill_and_outline(tmp_path):
    mesh = disk_mesh()
    filled = tmp_path / "filled.png"
    outline = tmp_path / "outline.png"
    mesh.draw(True, "orangered", save_name=str(filled), show=False)
    mesh.draw(fill_inside=False, edge_color="black", save_name=str(outline), show=False)
    assert filled.stat().st_size > 0
    assert outline.stat().st_size > 0


def test_draw_clips_out_of_range_properties(tmp_path):
    mesh = disk_mesh()
    for element in mesh.elements:
        element.element_property = 300.0
    path = tmp_path / "clipped.png"
    mesh.draw(show=False, save_name=str(path))
    assert path.stat().st_size > 0


def test_draw_draws_all_elements():
    import matplotlib.pyplot as plt

    mesh = disk_mesh()
    fig = mesh.draw(show=False)
    ax = fig.axes[0]
    collections = ax.collections
    assert len(collections) == 1
    vertices = collections[0].get_paths()
    assert len(vertices) == len(mesh.elements)
    plt.close(fig)
