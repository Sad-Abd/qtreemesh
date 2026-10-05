"""Regression tests freezing the mesh behavior against stored references.

The reference meshes in ``tests/data/reference_meshes.json`` were generated
from the quadtree implementation they freeze; any change to splitting,
balancing, labeling or hanging-node handling must reproduce them exactly.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qtreemesh import QTree, QTreeMesh, image_preprocess

DATA = Path(__file__).parent / "data" / "reference_meshes.json"


def build(array, crit, max_size=None, balancing=True):
    quad = QTree(None, image_preprocess(array), crit, max_size=max_size)
    mesh = QTreeMesh(quad, balancing=balancing)
    mesh.create_elements()
    return mesh


def mesh_state(mesh):
    return {
        "n_nodes": int(mesh.nodes.shape[0]),
        "nodes": [[round(float(x), 9) for x in row] for row in mesh.nodes],
        "elements": [
            {
                "number": int(e.number),
                "nodes": [int(n) for n in e.nodes_numbers],
                "type": [int(e.element_type[0]), int(e.element_type[1]), int(e.element_type[2])],
                "property": round(float(e.element_property), 12),
                "coords": [[round(float(x), 9) for x in p] for p in e.nodes_coordinates],
            }
            for e in mesh.elements
        ],
    }


def case_images():
    rng = np.random.default_rng(7)
    size = 64
    yy, xx = np.mgrid[0:size, 0:size]
    disk = np.zeros((size, size), dtype=int)
    disk[((xx - size // 2) ** 2 + (yy - size // 2) ** 2) < size**2 // 16] = 10
    two = np.zeros((32, 32), dtype=int)
    two[10:20, 10:22] = 5
    return {
        "disk64": disk,
        "uniform_nobalance": np.zeros((32, 32), dtype=int),
        "twophase32": two,
        "grad64": (xx + yy).astype(int),
        "noisy32": rng.integers(0, 4, size=(32, 32)),
        "rect20x37": rng.integers(0, 3, size=(20, 37)),
    }


def case_build(name):
    images = case_images()
    if name == "seven_crit125":
        im = np.asarray(
            Image.open(Path(__file__).parents[1] / "examples" / "7.jpg").convert("L")
        )
        return build(im, 125)
    if name == "seven_crit125_max16":
        im = np.asarray(
            Image.open(Path(__file__).parents[1] / "examples" / "7.jpg").convert("L")
        )
        return build(im, 125, max_size=16)
    if name == "seven_crit30":
        im = np.asarray(
            Image.open(Path(__file__).parents[1] / "examples" / "7.jpg").convert("L")
        )
        return build(im, 30)
    if name == "disk64":
        return build(images["disk64"], 1)
    if name == "uniform_nobalance":
        return build(images["uniform_nobalance"], 1, balancing=False)
    if name == "twophase32":
        return build(images["twophase32"], 1)
    if name == "grad64":
        return build(images["grad64"], 2)
    if name == "noisy32":
        return build(images["noisy32"], 1)
    if name == "noisy32_max8":
        return build(images["noisy32"], 1, max_size=8)
    if name == "rect20x37":
        return build(images["rect20x37"], 1)
    raise KeyError(name)


@pytest.mark.parametrize(
    "name",
    [
        "seven_crit125",
        "seven_crit125_max16",
        "seven_crit30",
        "disk64",
        "uniform_nobalance",
        "twophase32",
        "grad64",
        "noisy32",
        "noisy32_max8",
        "rect20x37",
    ],
)
def test_mesh_matches_reference(name):
    references = json.loads(DATA.read_text())
    mesh = case_build(name)
    assert mesh_state(mesh) == references[name]


def test_pyramid_lookup_matches_direct_scan():
    rng = np.random.default_rng(11)
    # smooth regions plus a sharp inclusion: every level of the tree is exercised
    array = rng.integers(0, 3, size=(32, 32))
    array[8:24, 8:24] = 20
    array[12:16, 12:16] = 30
    quad = QTree(None, array, 1)

    def walk(node):
        yield node
        if node.divided:
            for child in (
                node.north_west,
                node.north_east,
                node.south_west,
                node.south_east,
            ):
                yield from walk(child)

    for node in walk(quad):
        assert node._cell_range() == int(np.max(node.array) - np.min(node.array))


def test_pyramid_fallback_for_non_square_arrays():
    array = np.zeros((4, 16), dtype=int)
    array[:, 8:] = 9  # splits once into four homogeneous leaves
    quad = QTree(None, array, 1)
    assert quad._pyramid_source._block_max is None
    mesh = QTreeMesh(quad)
    mesh.create_elements()
    assert len(mesh.elements) == 4


def test_property_is_lazy_but_identical():
    array = np.zeros((16, 16), dtype=int)
    array[4:8, 4:8] = 6
    quad = QTree(None, array, 1)
    assert quad._property is None  # not computed during construction
    assert quad.property == np.mean(array)
    leaf = quad.save_leaves()[0]
    assert leaf.property == np.mean(leaf.array)
    quad.property = 42.0
    assert quad.property == 42.0


def test_property_descriptor_class_access():
    from qtreemesh._qtreemesh import _LazyProperty

    assert isinstance(QTree.property, _LazyProperty)
