"""Tests for QTreeElement.quad_treatment and QTreeMesh.mode_detection.

mode_detection maps the four boolean edge flags (bottom, right, top, left,
counter-clockwise) to a basic mode and rotation. quad_treatment splits an
element with hanging nodes into triangles and quads. The exact-output cases
pin the current (verified) behavior as a regression net.
"""

import pytest

from qtreemesh import QTreeElement, QTreeMesh


@pytest.mark.parametrize(
    "mode,expected",
    [
        ([False, False, False, False], [1, 0]),
        ([True, False, False, False], [2, 0]),
        ([False, True, False, False], [2, 90]),
        ([False, False, True, False], [2, 180]),
        ([False, False, False, True], [2, 270]),
        ([True, True, False, False], [3, 0]),
        ([False, True, True, False], [3, 90]),
        ([False, False, True, True], [3, 180]),
        ([True, False, False, True], [3, 270]),
        ([True, False, True, False], [4, 0]),
        ([False, True, False, True], [4, 90]),
        ([False, True, True, True], [5, 0]),
        ([True, False, True, True], [5, 90]),
        ([True, True, False, True], [5, 180]),
        ([True, True, True, False], [5, 270]),
        ([True, True, True, True], [6, 0]),
    ],
)
def test_mode_detection_covers_all_edge_configurations(mode, expected):
    assert QTreeMesh.mode_detection(mode) == expected


def make_element(mode_number, rotation):
    node_counts = {1: 4, 2: 5, 3: 6, 4: 6, 5: 7, 6: 8}
    nodes = list(range(10, 10 + node_counts[mode_number]))
    return QTreeElement(1, nodes, [], [mode_number, rotation], 1.0)


# sub-element count per (basic mode, force_triangulate)
EXPECTED_COUNTS = {
    (1, True): 2,
    (1, False): 1,
    (2, True): 3,
    (2, False): 3,
    (3, True): 4,
    (3, False): 4,
    (4, True): 4,
    (4, False): 2,
    (5, True): 5,
    (5, False): 4,
    (6, True): 6,
    (6, False): 5,
}


@pytest.mark.parametrize("force", [True, False])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("mode_number", [1, 2, 3, 4, 5, 6])
def test_quad_treatment_structure(mode_number, rotation, force):
    element = make_element(mode_number, rotation)
    result = element.quad_treatment(force)
    assert len(result) == EXPECTED_COUNTS[(mode_number, force)]
    for sub_element in result:
        assert len(sub_element) in (3, 4)
        assert set(sub_element) <= set(element.nodes_numbers)


def test_quad_treatment_mode1():
    element = make_element(1, 0)
    assert element.quad_treatment(True) == [[10, 11, 12], [10, 12, 13]]
    assert element.quad_treatment(False) == [[10, 11, 12, 13]]


def test_quad_treatment_mode2_rotation_zero():
    element = make_element(2, 0)
    assert element.quad_treatment(True) == [
        [14, 10, 11],
        [14, 11, 13],
        [13, 11, 12],
    ]


def test_quad_treatment_mode2_rotation_ninety():
    element = make_element(2, 90)
    assert element.quad_treatment(True) == [
        [10, 11, 12],
        [10, 12, 14],
        [14, 12, 13],
    ]


def test_quad_treatment_mode4_keeps_two_quads():
    element = make_element(4, 0)
    assert element.quad_treatment(False) == [[10, 11, 14, 15], [11, 12, 13, 14]]


def test_quad_treatment_mode3_rotation_270_rolls_once_more():
    element = make_element(3, 270)
    assert element.quad_treatment(True) == [
        [13, 14, 15],
        [15, 10, 11],
        [11, 12, 13],
        [13, 15, 11],
    ]
