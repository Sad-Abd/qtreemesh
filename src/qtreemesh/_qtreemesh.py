"""
A module for generating quadtree mesh from an image.

The image is partitioned by a quadtree based on pixel intensities, the leaves
are converted to mesh elements (with hanging-node handling), and the mesh can
be exported to VTK or adjusted for FEM analysis.

Author : Sadjad Abedi
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection


def _max_adjacent_difference(array):
    """
    Maximum absolute difference between horizontally or vertically adjacent
    pixels of a 2D array. Zero when the array has a single row or column.
    """
    greatest = 0.0
    if array.shape[1] > 1:
        greatest = np.abs(np.diff(array, axis=1)).max()
    if array.shape[0] > 1:
        greatest = max(greatest, np.abs(np.diff(array, axis=0)).max())
    return greatest


class Point:
    """
    A class used to represent a Point.

    ...

    Attributes
    ----------
    x_coord : float
        Horizontal coordinate of the point
    y_coord : float
        Vertical coordinate of the point
    xy_coord : tuple (float, float)
        Horizontal and vertical coordinates of the point

    Methods
    -------
    coord_sum(second_coord)
        Return a Point object from summation of current Point coordinates
        with a second pair of coordinates.
    """

    def __init__(self, coord):
        self.x_coord = coord[0]
        self.y_coord = coord[1]
        self.xy_coord = coord

    def coord_sum(self, second_coord):
        """
        Return a Point object from summation of current Point coordinates
        with a second pair of coordinates.

        Parameters
        ----------
        second_coord : tuple
            The Second Tuple to Add.

        Returns
        -------
        new_point : tuple
            Point object.

        """
        new_point = Point(
            (self.x_coord + second_coord[0], self.y_coord + second_coord[1])
        )

        return new_point


class _LazyProperty:
    """
    Descriptor for the mean pixel intensity of a cell's array (`QTree.property`).

    The value is computed on first access and cached; assigning to the
    attribute stores the given value.
    """

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if obj._property is None:
            obj._property = np.mean(obj.array)
        return obj._property

    def __set__(self, obj, value):
        obj._property = value


class QTree:
    """
    A class used to represent a quadtree.

    ...

    Attributes
    ----------
    parent : None or QTree object
        Parent leaf of current leaf
    array : numpy array
        Corresponding partition of image array.
    crit : int, optional
        The criteria used for partitioning. A cell splits only when the
        difference between its maximum and minimum pixel intensities is
        strictly greater than crit; with the default crit = 1, two regions
        whose intensities differ by exactly 1 are not split.
        Default value is 1.
    max_size : int, optional
        Maximum cell size in pixels. A cell whose side (dimension) exceeds
        max_size is split even when its pixels are homogeneous, so uniform
        regions are not left as one huge cell. None (default) imposes no
        limit. Cells halve at each split, so the effective cell side is the
        largest power of 2 not exceeding max_size.
    label_mode : bool, optional
        Treat the array as a label map instead of an intensity image: a
        cell splits whenever it contains more than one distinct label,
        regardless of the numeric distance between labels. `crit` is not
        used and must stay at its default. Every cell of the resulting
        tree is label-homogeneous. Default False.
    grad_crit : float, optional
        Refine where the intensity changes steeply: a cell also splits when
        the maximum absolute difference between horizontally or vertically
        adjacent pixels inside it exceeds grad_crit, even if its intensity
        range satisfies `crit`. None (default) disables this criterion and
        the associated scan. Cannot be combined with label_mode.
    scale : float, optional
        The ratio between pixels units and real units. For example, when scale is 2,
        each pixel represents a 2*2 ($mm^2$ or $in^2$ or ...) square part of the object.
        This parameter is used to calculate spatial location of pixels.
        Default value is 1.
    bottom_left_corner : Point, optional
        A Point object that contains the coordinate of the bottom left corner of each
        partition. Default position is (0.0,0.0).
    top_right_corner : Point
        A Point object that contains the coordinate of the top right corner of each
        partition. Calculated from bottom_left_corner.
    dimension : int
        Dimension of the cell (number of pixels in each direction).
    depth : int, optional
        Depth of the node in tree. It's 0 for the root.
    property : float
        An indicator for material properties calculated by averaging
        the pixels intensities.
    divided : bool
        Indicate wether the cell is divided (is an inner node) or not (is a leaf).
    count_leaves : int
        Total number of external nodes (leaves).
    north_west : None or QTree object
        Address the cell located in northwest part of current cell. None for leaves.
    north_east : None or QTree object
        Address the cell located in northeast part of current cell. None for leaves.
    south_west : None or QTree object
        Address the cell located in southwest part of current cell. None for leaves.
    south_east : None or QTree object
        Address the cell located in southeast part of current cell. None for leaves.


    Methods
    -------
    sectors()
        A recursive function to partition the array and create subtrees.
    count_leaves()
        A property that returns the total number of external nodes (leaves).
    save_leaves()
        A method that returns a list of external nodes (leaves).
    north_neighbor()
        A recursive function that return north neighbor of the cell.
        return none if the cell is on the top of the image.
    south_neighbor()
        A recursive function that return south neighbor of the cell.
        return none if the cell is on the bottom of the image.
    west_neighbor()
        A recursive function that return west neighbor of the cell.
        return none if the cell is on the left side of the image.
    east_neighbor()
        A recursive function that return east neighbor of the cell.
        return none if the cell is on the right side of the image.
    need_split(node):
        Check 4 sides neighbors for more than 2:1 ratio. Return True
        if the cell has to be splitted for 2:1 balancing.
    balancing():
        Balance QTree for 2:1 ratio.

    """

    def __init__(
        self,
        parent,
        array,
        crit=1,
        scale=1.0,
        bottom_left_corner=Point((0.0, 0.0)),
        depth=0,
        max_size=None,
        label_mode=False,
        grad_crit=None,
    ):
        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be at least 1 (or None for no limit)")
        if label_mode and crit != 1:
            raise ValueError(
                "the split criterion is not used in label_mode; leave crit at 1"
            )
        if label_mode and grad_crit is not None:
            raise ValueError("grad_crit cannot be combined with label_mode")
        if grad_crit is not None and (not np.isfinite(grad_crit) or grad_crit <= 0):
            raise ValueError(f"grad_crit must be positive and finite, got {grad_crit}")
        if np.asarray(array).ndim != 2:
            raise ValueError(
                f"the image array must be 2D, got {np.asarray(array).ndim} dimensions"
            )
        self.north_west = None  # NorthWest Section Initiated Empty
        self.north_east = None  # NorthEast Section Initiated Empty
        self.south_west = None  # SouthWest Section Initiated Empty
        self.south_east = None  # SouthEast Section Initiated Empty
        self.parent = parent
        self.array = array
        self.divided = False
        self.depth = depth
        self.crit = crit
        self.scale = scale
        self.max_size = max_size
        self.label_mode = label_mode
        self.grad_crit = grad_crit
        self._pyramid_source = parent._pyramid_source if parent is not None else self
        self._block_max = None
        self._block_min = None
        self._pyramid_failed = False
        self._property = None
        self.bottom_left_corner = bottom_left_corner  # BottomLeft Coordinates
        self.top_right_corner = bottom_left_corner.coord_sum(
            (array.shape[1] * scale, array.shape[0] * scale)
        )  # TopRight Coordinates
        self.dimension = np.sqrt(array.size)  # To define scale requirement

        # SPLITTING
        if label_mode:
            split = self._cell_range() != 0
        else:
            split = self._cell_range() > crit
            if not split and self.grad_crit is not None:
                split = _max_adjacent_difference(array) > self.grad_crit
        if split or (
            self.max_size is not None and self.dimension > self.max_size
        ):
            self.sectors()

    def _block_indices(self):
        """
        Position (row, column) of the cell within its level of the pyramid.

        Pyramid rows count from the top of the root array, while the y
        coordinate grows upwards, so the row index follows from the cell's
        top edge.
        """
        side = self.dimension
        top = self.bottom_left_corner.y_coord / self.scale + side
        cols = self.bottom_left_corner.x_coord / self.scale
        root_side = self._pyramid_source.dimension
        return int(round((root_side - top) / side)), int(round(cols / side))

    def _build_pyramids(self):
        """
        Per-level blockwise max/min tables of the root array.

        Entry k of each list holds, for every 2^k x 2^k block of the root
        array, its maximum (minimum). The tables are exact, so a lookup
        returns the same value as np.max (np.min) on the cell's array. They
        are built only when the root array is a square with a power-of-2
        side; otherwise the flags mark the direct computation as fallback.
        """
        n = self.array.shape[0]
        if self.array.shape[0] != self.array.shape[1] or n < 2 or n & (n - 1) != 0:
            self._pyramid_failed = True
            return
        block_max = [self.array]
        block_min = [self.array]
        while block_max[-1].shape[0] > 1:
            upper_max = block_max[-1]
            upper_min = block_min[-1]
            block_max.append(
                np.maximum(
                    np.maximum(upper_max[0::2, 0::2], upper_max[0::2, 1::2]),
                    np.maximum(upper_max[1::2, 0::2], upper_max[1::2, 1::2]),
                )
            )
            block_min.append(
                np.minimum(
                    np.minimum(upper_min[0::2, 0::2], upper_min[0::2, 1::2]),
                    np.minimum(upper_min[1::2, 0::2], upper_min[1::2, 1::2]),
                )
            )
        self._block_max = block_max
        self._block_min = block_min

    def _cell_range(self):
        """
        Difference between the maximum and minimum of the cell's array.

        Uses the level pyramids of the root array when available, otherwise
        scans the array directly.
        """
        source = self._pyramid_source
        if source._block_max is None and not source._pyramid_failed:
            source._build_pyramids()
        if source._block_max is None:
            return np.max(self.array) - np.min(self.array)
        level = int(round(np.log2(self.dimension)))
        row, col = self._block_indices()
        return int(source._block_max[level][row, col]) - int(
            source._block_min[level][row, col]
        )

    def sectors(self):
        """
        A recursive function to create subtrees.

        Returns
        -------
        None.

        """
        self.divided = True
        size = self.array.shape
        bottom_left_north_west = self.bottom_left_corner.coord_sum(
            (0, (size[0] / 2) * self.scale)
        )
        self.north_west = QTree(
            self,
            self.array[0 : size[0] // 2, 0 : size[1] // 2],
            self.crit,
            self.scale,
            bottom_left_north_west,
            self.depth + 1,
            self.max_size,
            self.label_mode,
            self.grad_crit,
        )
        bottom_left_north_east = self.bottom_left_corner.coord_sum(
            ((size[1] / 2) * self.scale, (size[0] / 2) * self.scale)
        )
        self.north_east = QTree(
            self,
            self.array[0 : size[0] // 2, size[1] // 2 : size[1]],
            self.crit,
            self.scale,
            bottom_left_north_east,
            self.depth + 1,
            self.max_size,
            self.label_mode,
            self.grad_crit,
        )
        bottom_left_south_west = self.bottom_left_corner
        self.south_west = QTree(
            self,
            self.array[size[0] // 2 : size[0], 0 : size[1] // 2],
            self.crit,
            self.scale,
            bottom_left_south_west,
            self.depth + 1,
            self.max_size,
            self.label_mode,
            self.grad_crit,
        )
        bottom_left_south_east = self.bottom_left_corner.coord_sum(
            ((size[1] / 2) * self.scale, 0)
        )
        self.south_east = QTree(
            self,
            self.array[size[0] // 2 : size[0], size[1] // 2 : size[1]],
            self.crit,
            self.scale,
            bottom_left_south_east,
            self.depth + 1,
            self.max_size,
            self.label_mode,
            self.grad_crit,
        )

    @property
    def count_leaves(self):
        """
        A property that recursively calculates the number of tree leaves.

        Returns
        -------
        int
            Summation of leaves of subtrees.

        """
        # If the node itself is "None" there are no leaves ->  return 0
        if self is None:
            return 0

        # If the node is a leaf then children will be "None" -> return 1
        if (
            self.north_west is None
            and self.north_east is None
            and self.south_west is None
            and self.south_east is None
        ):
            return 1

        # Now we count the leaves in subtrees and return the sum
        return (
            self.north_west.count_leaves
            + self.north_east.count_leaves
            + self.south_west.count_leaves
            + self.south_east.count_leaves
        )

    def save_leaves(self):
        """
        A function that stores all the leaves.

        Returns
        -------
        leaves_list : list
            A list of all external nodes (leaves) of the tree.
        """

        # Stack to store all the nodes of tree
        node_list = []

        # Stack to store all the leaf nodes
        leaves_list = []

        # Push the root node
        node_list.append(self)

        while len(node_list) != 0:
            curr = node_list.pop()

            # If current node has a child push it onto the first stack
            if curr.divided:
                node_list.append(curr.north_west)
                node_list.append(curr.north_east)
                node_list.append(curr.south_west)
                node_list.append(curr.south_east)
            # If current node is a leaf node push it onto the second stack
            else:
                leaves_list.append(curr)

        return leaves_list

    def north_neighbor(self):
        """
        Find north neighbor of Node.

        Returns
        -------
            Qtree object of north neighbor of Node.
        """
        if self.parent is None:
            return None
        if self == self.parent.south_west:
            return self.parent.north_west
        if self == self.parent.south_east:
            return self.parent.north_east
        parent_neighbor = self.parent.north_neighbor()
        if parent_neighbor is None:
            return parent_neighbor
        elif not parent_neighbor.divided:
            return parent_neighbor
        elif self == self.parent.north_west:
            return parent_neighbor.south_west
        else:
            return parent_neighbor.south_east

    def south_neighbor(self):
        """
        Find south neighbor of Node.

        Returns
        -------
            Qtree object of south neighbor of Node.
        """
        if self.parent is None:
            return None
        if self == self.parent.north_west:
            return self.parent.south_west
        if self == self.parent.north_east:
            return self.parent.south_east
        parent_neighbor = self.parent.south_neighbor()
        if parent_neighbor is None:
            return parent_neighbor
        elif not parent_neighbor.divided:
            return parent_neighbor
        elif self == self.parent.south_west:
            return parent_neighbor.north_west
        else:
            return parent_neighbor.north_east

    def west_neighbor(self):
        """
        Find west neighbor of Node.

        Returns
        -------
            Qtree object of west neighbor of Node.
        """
        if self.parent is None:
            return None
        if self == self.parent.north_east:
            return self.parent.north_west
        if self == self.parent.south_east:
            return self.parent.south_west

        parent_neighbor = self.parent.west_neighbor()
        if parent_neighbor is None:
            return parent_neighbor
        elif not parent_neighbor.divided:
            return parent_neighbor
        elif self == self.parent.south_west:
            return parent_neighbor.south_east
        else:
            return parent_neighbor.north_east

    def east_neighbor(self):
        """
        Find east neighbor of Node.

        Returns
        -------
            Qtree object of east neighbor of Node.
        """
        if self.parent is None:
            return None
        if self == self.parent.north_west:
            return self.parent.north_east
        if self == self.parent.south_west:
            return self.parent.south_east

        parent_neighbor = self.parent.east_neighbor()
        if parent_neighbor is None:
            return parent_neighbor
        elif not parent_neighbor.divided:
            return parent_neighbor
        elif self == self.parent.south_east:
            return parent_neighbor.south_west
        else:
            return parent_neighbor.north_west

    @staticmethod
    def need_split(node):
        """
        Check 4 sides neighbors for more than 2:1 ratio. Return True
        if the cell has to be splitted for 2:1 balancing.

        Returns
        -------
            boolean
        """
        if node is None:
            return False
        if node.north_neighbor() is not None:
            if node.north_neighbor().divided:
                if (
                    node.north_neighbor().south_west.divided
                    or node.north_neighbor().south_east.divided
                ):
                    return True
        if node.south_neighbor() is not None:
            if node.south_neighbor().divided:
                if (
                    node.south_neighbor().north_west.divided
                    or node.south_neighbor().north_east.divided
                ):
                    return True
        if node.west_neighbor() is not None:
            if node.west_neighbor().divided:
                if (
                    node.west_neighbor().north_east.divided
                    or node.west_neighbor().south_east.divided
                ):
                    return True
        if node.east_neighbor() is not None:
            if node.east_neighbor().divided:
                if (
                    node.east_neighbor().north_west.divided
                    or node.east_neighbor().south_west.divided
                ):
                    return True

        return False

    def balancing(self):
        """
        Balance QTree for 2:1 ratio.

        Returns
        -------
        None.
        """
        leaves = self.save_leaves()
        while len(leaves) != 0:
            node = leaves.pop()
            if not node.divided:
                if self.need_split(node):
                    node.sectors()
                    leaves.extend(
                        [
                            node.south_west,
                            node.south_east,
                            node.north_west,
                            node.north_east,
                        ]
                    )
                    if self.need_split(node.north_neighbor()):
                        leaves.append(node.north_neighbor())
                    if self.need_split(node.south_neighbor()):
                        leaves.append(node.south_neighbor())
                    if self.need_split(node.west_neighbor()):
                        leaves.append(node.west_neighbor())
                    if self.need_split(node.east_neighbor()):
                        leaves.append(node.east_neighbor())

    # placed after every @property use in this class body: the name
    # `property` must still resolve to the builtin at those lines
    property = _LazyProperty()


class QTreeElement:
    """
    A class used to represent a quadtree element.

    ...

    Attributes
    ----------
    number : int
        Element number label.
    nodes_numbers : list(int)
        Number of nodes of the element. The order of nodes is counterclockwise.
    nodes_coordinates : list(tuple)
        A list that contains a tuple of x-y coordinates of the nodes
        based on nodes_numbers.
    element_type : list()
        A list that contains three values. First value indicate the element
        mode based on basic modes (1 to 6). The second value indicate the
        angle of rotation that element needs to convert to basic modes. The
        third value indicate scale parameter. [Used for SBFEM mesh]
    element_property : float
        Element indicator of material properties calculated by averaging
        the pixels intensities.
    corner_numbers : list(int)
        Numbers of the four corner nodes, set by `QTreeMesh.create_elements`.
        The order is counter-clockwise starting from the south-west corner;
        unlike `nodes_numbers` it never contains hanging nodes.
    hanging_nodes : list(tuple (int, int, int))
        One entry per hanging node on the element edges:
        (hanging node, first edge corner, second edge corner), set by
        `QTreeMesh.create_elements`. Empty for elements without hanging
        nodes.



    Methods
    -------
    quad_treatment(force_triangulate=False)
        Modify a quadtree element by handling hanging nodes
    """

    def __init__(
        self, label, nodes_numbers, nodes_coordinates, element_type, element_property
    ) -> None:
        self.number = label
        self.nodes_numbers = nodes_numbers
        self.nodes_coordinates = nodes_coordinates
        self.element_type = element_type
        self.element_property = element_property

    def quad_treatment(self, force_triangulate=True):
        """
        Modify a quadtree element by handling hanging nodes.

        This function processes a quadtree element to handle hanging nodes. It returns the
        modified node numbers for the mesh.

        Parameters:
        -----------
        force_triangulate : bool, optional (default=True)
            If True, forces triangulation when applicable.

        Returns:
        --------
        new_nodes_numbers : list
            A list containing the modified node numbers for the mesh, following
            the specified treatment.
        """

        def roll_list_left(lst, positions=1):
            positions = positions % len(lst)
            return lst[positions:] + lst[:positions]

        new_nodes_numbers = []
        if self.element_type[0] == 1:
            if force_triangulate:
                new_nodes_numbers = [
                    [self.nodes_numbers[i] for i in [0, 1, 2]],
                    [self.nodes_numbers[i] for i in [0, 2, 3]],
                ]
            else:
                new_nodes_numbers = [self.nodes_numbers]
        elif self.element_type[0] == 2:
            rotated_indices = roll_list_left(
                self.nodes_numbers, positions=self.element_type[1] // 90
            )
            new_nodes_numbers = [
                [rotated_indices[i] for i in [4, 0, 1]],
                [rotated_indices[i] for i in [4, 1, 3]],
                [rotated_indices[i] for i in [3, 1, 2]],
            ]
        elif self.element_type[0] == 3:
            t = 0
            if self.element_type[1] == 270:
                t += 1
            rotated_indices = roll_list_left(
                self.nodes_numbers, positions=self.element_type[1] // 90 + t
            )
            new_nodes_numbers = [
                [rotated_indices[i] for i in [5, 0, 1]],
                [rotated_indices[i] for i in [1, 2, 3]],
                [rotated_indices[i] for i in [3, 4, 5]],
                [rotated_indices[i] for i in [5, 1, 3]],
            ]
        elif self.element_type[0] == 4:
            if force_triangulate:
                rotated_indices = roll_list_left(
                    self.nodes_numbers, positions=self.element_type[1] // 90
                )
                new_nodes_numbers = [
                    [rotated_indices[i] for i in [5, 0, 1]],
                    [rotated_indices[i] for i in [1, 2, 3]],
                    [rotated_indices[i] for i in [4, 1, 3]],
                    [rotated_indices[i] for i in [5, 1, 4]],
                ]
            else:
                rotated_indices = roll_list_left(
                    self.nodes_numbers, positions=self.element_type[1] // 90
                )
                new_nodes_numbers = [
                    [rotated_indices[i] for i in [0, 1, 4, 5]],
                    [rotated_indices[i] for i in [1, 2, 3, 4]],
                ]
        elif self.element_type[0] == 5:
            if force_triangulate:
                rotated_indices = roll_list_left(
                    self.nodes_numbers, positions=2 * self.element_type[1] // 90
                )
                new_nodes_numbers = [
                    [rotated_indices[i] for i in [6, 4, 5]],
                    [rotated_indices[i] for i in [4, 6, 2]],
                    [rotated_indices[i] for i in [2, 3, 4]],
                    [rotated_indices[i] for i in [0, 1, 2]],
                    [rotated_indices[i] for i in [0, 2, 6]],
                ]
            else:
                rotated_indices = roll_list_left(
                    self.nodes_numbers, positions=2 * self.element_type[1] // 90
                )
                new_nodes_numbers = [
                    [rotated_indices[i] for i in [6, 4, 5]],
                    [rotated_indices[i] for i in [4, 6, 2]],
                    [rotated_indices[i] for i in [2, 3, 4]],
                    [rotated_indices[i] for i in [0, 1, 2, 6]],
                ]
        else:
            if force_triangulate:
                new_nodes_numbers = [
                    [self.nodes_numbers[i] for i in [7, 0, 1]],
                    [self.nodes_numbers[i] for i in [1, 2, 3]],
                    [self.nodes_numbers[i] for i in [3, 4, 5]],
                    [self.nodes_numbers[i] for i in [5, 6, 7]],
                    [self.nodes_numbers[i] for i in [7, 1, 3]],
                    [self.nodes_numbers[i] for i in [7, 3, 5]],
                ]
            else:
                new_nodes_numbers = [
                    [self.nodes_numbers[i] for i in [7, 0, 1]],
                    [self.nodes_numbers[i] for i in [1, 2, 3]],
                    [self.nodes_numbers[i] for i in [3, 4, 5]],
                    [self.nodes_numbers[i] for i in [5, 6, 7]],
                    [self.nodes_numbers[i] for i in [7, 1, 3, 5]],
                ]

        return new_nodes_numbers


def _write_scalar_block(file_open, name, values):
    """Write one legacy-VTK scalar array block for cell data."""
    file_open.write(f"SCALARS {name} float 1 \nLOOKUP_TABLE default \n")
    for item in values:
        file_open.write(f"{item}\n")


class QTreeMesh:
    """
    A class used to represent a quadtree mesh.

    ...

    Attributes
    ----------
    quad_tree : QTree object
        The main quad-tree structure from which initial mesh is generated.
    balancing : bool, optional
        Indicate whether the quad-tree is balanced for 2:1 ratio or not.
    leaves : list
        Outer nodes of the quad-tree.
    elements : list
        List of mesh elements as QTreeElement objects.
    nodes : list
        List of coordinates of mesh nodes.
    content_shape : None or tuple (rows, cols)
        Shape of the original image before padding, set by `trim_padding`.
        None while the mesh covers the whole padded image.



    Methods
    -------
    from_image()
        Build a mesh from an image array in one step.
    create_elements()
        Generate elements from cells in quad-tree.
    labeling()
        Labeling cells and their corner points.
    refactor_edge()
        Considering edge points in the cells attributes and detect cell modes
        based on the presence and location of edge points.
    mode_detection()
        Detect cell modes based on the presence and location of edge points.
    draw()
        Draw the generated mesh.
    vtk_export()
        Export mesh as unstructured grid in vtk file.
    adjust_mesh_for_FEM()
        Adjust the quadtree mesh for Finite Element Method (FEM) simulations.
    constrained_quads()
        Mesh of quadrilateral elements with hanging nodes as constraints.
    trim_padding()
        Remove elements that lie entirely in the padded region of the image.
    boundary_edges()
        Edges of the mesh that belong to a single element.
    boundary_nodes()
        Node numbers lying on the boundary edges of the mesh.
    pixel_to_element()
        Map every pixel to the 1-based number of the element covering it.
    element_labels()
        Exact per-cell label of every element from the quadtree image.
    """

    def __init__(self, quad_tree: QTree, balancing=True) -> None:
        self.quad_tree = quad_tree
        if balancing:
            self.quad_tree.balancing()
        self.leaves = self.quad_tree.save_leaves()

        self.elements = []
        self.nodes = None
        self.content_shape = None

    @classmethod
    def from_image(
        cls,
        image,
        crit=1,
        scale=1.0,
        max_size=None,
        balancing=True,
        label_mode=False,
        grad_crit=None,
    ):
        """
        Build a mesh from an image array in one step.

        The image is padded with `image_preprocess` to a square whose side is
        a power of 2, a quadtree is grown on the padded image, and the mesh
        elements are generated. Use `trim_padding` to drop the elements of
        the padded region. For label maps pass `label_mode=True` so that
        every cell is split until it holds a single label.

        Parameters
        ----------
        image : numpy array
            2D array of pixel intensities (or labels when `label_mode` is
            True).
        crit : int, optional
            Splitting criterion, as in `QTree`. Default 1.
        scale : float, optional
            Ratio between pixel units and real units, as in `QTree`.
            Default 1.0.
        max_size : int, optional
            Maximum cell size in pixels, as in `QTree`. Default None.
        balancing : bool, optional
            Whether to balance the quad-tree for a 2:1 ratio. Default True.
        label_mode : bool, optional
            Treat the array as a label map, as in `QTree`. Default False.
        grad_crit : float, optional
            Refine where the intensity changes steeply, as in `QTree`.
            Default None.

        Returns
        -------
        QTreeMesh
            Mesh with elements generated.
        """
        quad = QTree(
            None,
            image_preprocess(np.asarray(image)),
            crit,
            scale=scale,
            max_size=max_size,
            label_mode=label_mode,
            grad_crit=grad_crit,
        )
        mesh = cls(quad, balancing=balancing)
        mesh.create_elements()
        return mesh

    def create_elements(self):
        """
        Generate mesh elements from the quad-tree leaves.

        Labels every leaf cell and its corner nodes (labeling), inserts the
        hanging nodes present on cell edges and detects the element type
        (refactor_edge), then appends one QTreeElement per leaf to elements.
        """
        self.labeling()
        self.refactor_edge()
        for leaf in self.leaves:
            label = leaf.cell_number
            node_number = leaf.edge_points_numbers
            node_coordinate = [self.nodes[n - 1, :] for n in node_number]
            element_type = leaf.cell_type
            element_property = leaf.property
            element = QTreeElement(
                label, node_number, node_coordinate, element_type, element_property
            )
            element.corner_numbers = leaf.corner_numbers
            element.hanging_nodes = leaf.hanging_nodes
            self.elements.append(element)

    def labeling(self):
        """
        Label all cells and their corner points, and add corner points to mesh nodes.

        Cells are numbered 1-based in leaf traversal order; node numbers are
        1-based in first-seen order, so node k is stored at self.nodes[k - 1].
        """
        index = {}
        coords = []

        for label, leaf in enumerate(self.leaves, start=1):
            bottom_left = leaf.bottom_left_corner
            top_right = leaf.top_right_corner
            corners = (
                (bottom_left.x_coord, bottom_left.y_coord),
                (top_right.x_coord, bottom_left.y_coord),
                (top_right.x_coord, top_right.y_coord),
                (bottom_left.x_coord, top_right.y_coord),
            )
            leaf.edge_points_numbers = []
            leaf.nodes_coordinate = []
            for corner in corners:
                leaf.nodes_coordinate.append(np.array(corner))
                key = (float(corner[0]), float(corner[1]))
                number = index.get(key)
                if number is None:
                    coords.append(key)
                    number = len(coords)
                    index[key] = number
                leaf.edge_points_numbers.append(number)
            leaf.cell_number = label
        self.nodes = np.array(coords, dtype=float).reshape(-1, 2)

    def refactor_edge(self):
        """
        A function that consider edge points, add them to
        cells attributes, and detect cell modes based on the
        presence and location of edge points

        Raises
        ------
        ValueError
            If the quadtree is not balanced for a 2:1 ratio, since the
            hanging nodes cannot be located then.
        """

        def top_right_finder(edge_nums):
            node_numbers = [n - 1 for n in edge_nums]
            top_right_node_index = np.lexsort(
                (self.nodes[node_numbers][:, 0], self.nodes[node_numbers][:, 1])
            )
            return edge_nums[top_right_node_index[-1]]

        for leaf in self.leaves:
            newedge = list()
            mode = list()
            corners = list(leaf.edge_points_numbers)
            hanging = list()

            newedge.append(leaf.edge_points_numbers[0])
            if leaf.south_neighbor() is not None:
                if leaf.south_neighbor().divided:
                    if leaf.south_neighbor().north_west.divided:
                        raise ValueError(
                            "the quadtree is not balanced for a 2:1 ratio; "
                            "build QTreeMesh with balancing=True"
                        )
                    node = top_right_finder(
                        leaf.south_neighbor().north_west.edge_points_numbers
                    )
                    newedge.append(node)
                    hanging.append((node, corners[0], corners[1]))
                    mode.append(True)
                else:
                    mode.append(False)
            else:
                mode.append(False)

            newedge.append(leaf.edge_points_numbers[1])
            if leaf.east_neighbor() is not None:
                if leaf.east_neighbor().divided:
                    if leaf.east_neighbor().north_west.divided:
                        raise ValueError(
                            "the quadtree is not balanced for a 2:1 ratio; "
                            "build QTreeMesh with balancing=True"
                        )
                    node = leaf.east_neighbor().north_west.edge_points_numbers[0]
                    newedge.append(node)
                    hanging.append((node, corners[1], corners[2]))
                    mode.append(True)
                else:
                    mode.append(False)
            else:
                mode.append(False)

            newedge.append(leaf.edge_points_numbers[2])
            if leaf.north_neighbor() is not None:
                if leaf.north_neighbor().divided:
                    if leaf.north_neighbor().south_east.divided:
                        raise ValueError(
                            "the quadtree is not balanced for a 2:1 ratio; "
                            "build QTreeMesh with balancing=True"
                        )
                    node = leaf.north_neighbor().south_east.edge_points_numbers[0]
                    newedge.append(node)
                    hanging.append((node, corners[2], corners[3]))
                    mode.append(True)
                else:
                    mode.append(False)
            else:
                mode.append(False)

            newedge.append(leaf.edge_points_numbers[3])
            if leaf.west_neighbor() is not None:
                if leaf.west_neighbor().divided:
                    if leaf.west_neighbor().south_east.divided:
                        raise ValueError(
                            "the quadtree is not balanced for a 2:1 ratio; "
                            "build QTreeMesh with balancing=True"
                        )
                    node = top_right_finder(
                        leaf.west_neighbor().south_east.edge_points_numbers
                    )
                    newedge.append(node)
                    hanging.append((node, corners[3], corners[0]))
                    mode.append(True)
                else:
                    mode.append(False)
            else:
                mode.append(False)

            cell_type = self.mode_detection(mode)
            cell_type.append(leaf.dimension)
            leaf.edge_points_numbers = newedge
            leaf.cell_type = cell_type
            leaf.corner_numbers = corners
            leaf.hanging_nodes = hanging

    @staticmethod
    def mode_detection(mode):
        """
        A function that detect cell modes based on the
        presence and location of edge points.

        Basic modes:
         *---* *---* *---*
         |   | |   | |   |
         | 1 | | 2 | | 3 *
         |   | |   | |   |
         *---* *-*-* *-*-*
         *-*-* *-*-* *-*-*
         |   | |   | |   |
         | 4 | * 5 * * 6 *
         |   | |   | |   |
         *-*-* *---* *-*-*

        Parameters
        ----------
        mode : list
            A list of booleans that indicates the presence of
            the edge node on each edge, starting from bottom edge
            and rotating counter-clockwise.


        Returns
        -------
        _ : list
            A list that first index determine basic mode number and
            the second index determine the angle of rotation needed
            to acquire the basic mode.
        """

        number_edge_points = mode.count(True)
        if number_edge_points == 0:
            return [1, 0]
        elif number_edge_points == 1:
            return [2, mode.index(True) * 90]
        elif number_edge_points == 2:
            if mode == [True, True, False, False]:
                return [3, 0]
            elif mode == [False, True, True, False]:
                return [3, 90]
            elif mode == [False, False, True, True]:
                return [3, 180]
            elif mode == [True, False, False, True]:
                return [3, 270]
            elif mode == [True, False, True, False]:
                return [4, 0]
            elif mode == [False, True, False, True]:
                return [4, 90]
        elif number_edge_points == 3:
            return [5, mode.index(False) * 90]
        elif number_edge_points == 4:
            return [6, 0]

    def draw(self, fill_inside=True, edge_color=None, save_name=None, show=True):
        """
        Draw elements with filling inside.

        All elements are drawn as a single polygon collection. When
        `fill_inside` is True, each element is filled with the grayscale
        value of its property divided by 255, clipped to [0, 1]. The figure
        is saved before it is shown.

        Parameters
        ----------
        fill_inside : bool, optional
            Fill elements with grayscale color based on
            element property.
        edge_color : None/str, optional
            The color for element edges.
        save_name : None/str, optional
            name of file to save figure.
        show : bool, optional
            Whether to display the figure. Default True.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.

        """
        fig = plt.figure(figsize=(10, 10), frameon=False)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axis_off()
        polygons = [
            np.asarray(element.nodes_coordinates, dtype=float)
            for element in self.elements
        ]
        if fill_inside:
            gray = np.clip(
                np.array(
                    [element.element_property for element in self.elements],
                    dtype=float,
                )
                / 255.0,
                0.0,
                1.0,
            )
            facecolors = np.column_stack([gray, gray, gray, np.ones_like(gray)])
        else:
            facecolors = "white"
        ax.add_collection(
            PolyCollection(polygons, facecolors=facecolors, edgecolors=edge_color)
        )
        ax.autoscale_view()
        fig.tight_layout()
        if save_name:
            fig.savefig(save_name)
        if show:
            plt.show()
        return fig

    def vtk_export(
        self,
        filename="output.vtk",
        adjusted=False,
        force_triangulation=True,
        cell_data=False,
    ):
        """
        Export mesh as unstructured grid to .vtk file.
        Creating the file is done manually, and no library is used.

        Parameters
        ----------
        filename : str, optional
            Output file name.
        adjusted : bool, optional
            If False (default), every element is exported as one polygon
            cell. If True, the elements are exported as treated by
            `quad_treatment`: triangles and/or quadrilaterals with their
            proper VTK cell types (5 and 9).
        force_triangulation : bool, optional
            Passed to `quad_treatment` when `adjusted` is True. Default True.
        cell_data : bool, optional
            If True, additional per-cell scalar arrays are written: the
            exact cell label (the minimum intensity for inhomogeneous
            cells), the basic mode number, the rotation angle and the cell
            size, followed by the element property. Default False.

        Returns
        -------
            None.

        """
        if adjusted:
            labels = self.element_labels(strict=False)
            cells, types, props = [], [], []
            label_data, mode_data, rotation_data, size_data = [], [], [], []
            for element, label in zip(self.elements, labels):
                connectivity = element.quad_treatment(force_triangulation)
                for sub_cells in connectivity:
                    cells.append(np.array(sub_cells) - 1)
                    types.append(5 if len(sub_cells) == 3 else 9)
                repeat = len(connectivity)
                props += [element.element_property] * repeat
                label_data += [label] * repeat
                mode_data += [element.element_type[0]] * repeat
                rotation_data += [element.element_type[1]] * repeat
                size_data += [element.element_type[2]] * repeat
        else:
            cells = [np.array(e.nodes_numbers) - 1 for e in self.elements]
            types = [7] * len(cells)
            props = [e.element_property for e in self.elements]
            label_data = list(self.element_labels(strict=False))
            mode_data = [e.element_type[0] for e in self.elements]
            rotation_data = [e.element_type[1] for e in self.elements]
            size_data = [e.element_type[2] for e in self.elements]

        file_open = open(filename, "w", encoding="utf-8")
        file_open.write("# vtk DataFile Version 2.0\nOutput Data\nASCII\n")
        file_open.write("DATASET UNSTRUCTURED_GRID\n")
        total_points = self.nodes.shape[0]
        file_open.write(f"POINTS {total_points} float\n")
        for each in self.nodes:
            file_open.write(f"{each[0]} {each[1]} 0.0\n")
        total_cells = len(cells)
        total_data = sum(i.shape[0] for i in cells) + total_cells
        file_open.write(f"CELLS {total_cells} {total_data}\n")
        for each in cells:
            file_open.write(f"{each.shape[0]} ")
            file_open.writelines(str(np.flip(each))[1:-1])
            file_open.write("\n")
        file_open.write(f"CELL_TYPES {total_cells}\n")
        for cell_type in types:
            file_open.write(f"{cell_type}\n")

        file_open.write(f"CELL_DATA {total_cells}\n")
        if cell_data:
            _write_scalar_block(file_open, "Label", label_data)
            _write_scalar_block(file_open, "Mode", mode_data)
            _write_scalar_block(file_open, "Rotation", rotation_data)
            _write_scalar_block(file_open, "Size", size_data)
        _write_scalar_block(file_open, "Average-Intensity", props)

        file_open.close()

    def adjust_mesh_for_FEM(self, force_triangulation=True):
        """
        Adjust the quadtree mesh for Finite Element Method (FEM) simulations.

        This method processes the quadtree mesh to make it suitable for FEM simulations by
        handling hanging nodes.

        Parameters:
        -----------
        force_triangulation : bool, optional
            If True, forces triangulation when applicable (default True).

        Returns:
        --------
        nodes : numpy array
            (x, y) coordinates of the mesh nodes.
        fem_elements : list of lists of int
            Node numbers of the adjusted elements.
        fem_properties : list of float
            Element properties calculated by averaging pixel intensities.
        """
        fem_elements = []
        fem_properties = []
        for element in self.elements:
            new_elements = element.quad_treatment(force_triangulation)
            fem_elements += new_elements
            fem_properties += [element.element_property] * len(new_elements)

        return self.nodes, fem_elements, fem_properties

    def constrained_quads(self):
        """
        Mesh of quadrilateral elements with hanging nodes as constraints.

        Unlike `adjust_mesh_for_FEM`, which removes hanging nodes by
        splitting elements, this method keeps one quadrilateral per mesh
        cell, built from its four corner nodes. The hanging nodes are
        returned separately as linear constraints: the displacement of the
        hanging node is the average of the displacements of the two corner
        nodes of the coarse edge it lies on, so it can be imposed as a
        multipoint constraint in a solver.

        Returns
        -------
        nodes : numpy array
            (x, y) coordinates of the mesh nodes.
        quad_elements : list of list of int
            Four corner node numbers per element (1-based, counter-
            clockwise).
        quad_properties : list of float
            Element properties calculated by averaging pixel intensities.
        constraints : list of tuple (int, int, int)
            (hanging node, corner node 1, corner node 2) per hanging node.
        """
        quad_elements = []
        quad_properties = []
        constraints = []
        for element in self.elements:
            quad_elements.append(list(element.corner_numbers))
            quad_properties.append(element.element_property)
            constraints += [tuple(node) for node in element.hanging_nodes]
        return self.nodes, quad_elements, quad_properties, constraints

    def trim_padding(self, rows, cols):
        """
        Remove elements that lie entirely in the padded region of the image.

        `image_preprocess` pads images with zero-intensity pixels to a square
        whose side is a power of 2, and the mesh covers the padded image too.
        This method deletes every element whose bounding box does not
        intersect the original image area, which spans `rows` rows and `cols`
        columns at the top-left corner of the padded array. Elements that
        straddle the boundary of the original area are kept.

        Element numbers, node numbers and the node array are left unchanged,
        so nodes may exist that no element uses afterwards.

        Parameters
        ----------
        rows, cols : int
            Number of rows and columns of the original image, before padding.

        Returns
        -------
        None.

        Raises
        ------
        ValueError
            If the original shape does not lie within the padded shape.
        """
        padded = self.quad_tree.array.shape
        if not (1 <= rows <= padded[0] and 1 <= cols <= padded[1]):
            raise ValueError(
                f"original shape ({rows}, {cols}) must lie within the padded "
                f"shape {padded}"
            )
        scale = self.quad_tree.scale
        x_edge = cols * scale
        y_edge = (padded[0] - rows) * scale
        tol = 1e-9 * scale
        elements, leaves = [], []
        for element, leaf in zip(self.elements, self.leaves):
            xy = np.asarray(element.nodes_coordinates, dtype=float)
            in_pad = xy[:, 0].min() >= x_edge - tol or xy[:, 1].max() <= y_edge + tol
            if not in_pad:
                elements.append(element)
                leaves.append(leaf)
        self.elements = elements
        self.leaves = leaves
        self.content_shape = (rows, cols)

    def boundary_edges(self):
        """
        Edges of the mesh that belong to a single element.

        Returns
        -------
        list of tuple
            (node_1, node_2) pairs of 1-based node numbers.
        """
        counts = {}
        for element in self.elements:
            nodes = element.nodes_numbers
            n = len(nodes)
            for i in range(n):
                a, b = int(nodes[i]), int(nodes[(i + 1) % n])
                key = (a, b) if a < b else (b, a)
                counts[key] = counts.get(key, 0) + 1
        return [edge for edge, count in counts.items() if count == 1]

    def boundary_nodes(self):
        """
        Node numbers lying on the boundary edges of the mesh.

        Returns
        -------
        list of int
            Sorted 1-based node numbers.
        """
        return sorted({node for edge in self.boundary_edges() for node in edge})

    def pixel_to_element(self):
        """
        Map every pixel of the preprocessed image to the element covering it.

        Returns
        -------
        pixel_elem : numpy array of int, shape (n_rows, n_cols)
            Entry [row, col] (row 0 = top of the image) holds the 1-based number
            of the element covering that pixel, so
            self.elements[pixel_elem[row, col] - 1] is the element itself.
            After `trim_padding`, pixels outside the original image area that
            no element covers hold 0.

        Raises
        ------
        RuntimeError
            If some pixels inside the original image area are not covered by
            any element (or, before `trim_padding`, if any pixel is uncovered).
        """
        scale = self.quad_tree.scale
        n_rows, n_cols = self.quad_tree.array.shape
        pixel_elem = -np.ones((n_rows, n_cols), dtype=int)
        for element in self.elements:
            xy = np.asarray(element.nodes_coordinates, dtype=float)
            lo = xy.min(axis=0)
            hi = xy.max(axis=0)
            c0 = int(round(lo[0] / scale))
            r0_bottom = int(round(lo[1] / scale))
            size_x = int(round((hi[0] - lo[0]) / scale))
            size_y = int(round((hi[1] - lo[1]) / scale))
            rows = slice(n_rows - r0_bottom - size_y, n_rows - r0_bottom)
            pixel_elem[rows, c0 : c0 + size_x] = element.number
        uncovered = pixel_elem < 0
        if uncovered.any():
            if self.content_shape is not None:
                rows, cols = self.content_shape
                outside = np.ones(pixel_elem.shape, dtype=bool)
                outside[:rows, :cols] = False
                uncovered &= ~outside
            if uncovered.any():
                raise RuntimeError("some pixels are not covered by a quadtree cell")
            pixel_elem[pixel_elem < 0] = 0
        return pixel_elem

    def element_labels(self, strict=True):
        """
        Exact per-cell label of every element, taken from the quadtree image.

        The label of an element is the single pixel intensity shared by all
        pixels under it, which - unlike the averaged element_property - is
        unambiguous for label and multi-material images.

        Parameters
        ----------
        strict : bool, optional
            If True (default), raise a ValueError when a cell spans multiple
            intensities, since no exact label exists for it. If False, such
            cells get the minimum intensity under them.

        Returns
        -------
        labels : numpy array
            One label per element, aligned with self.elements.
        """
        labels = []
        inhomogeneous = []
        for element, leaf in zip(self.elements, self.leaves):
            values = np.unique(leaf.array)
            if values.size == 1:
                labels.append(values.item())
            else:
                inhomogeneous.append(element.number)
                labels.append(values.min())
        if inhomogeneous and strict:
            raise ValueError(
                f"cells {inhomogeneous} span multiple intensities and have no "
                "exact label; reduce crit so every cell is homogeneous, or "
                "call with strict=False"
            )
        return np.asarray(labels)


def image_preprocess(image_array):
    """
    Pad the image to a square whose side is a power of 2.

    The image is first padded with zero-intensity rows or columns to a square,
    then padded the same way up to the next power of 2 on each side. The
    original pixels remain at the top-left of the result.

    Parameters
    ----------
    image_array : numpy array
        The array of the image.

    Returns
    -------
    image_array : numpy array
        The modified array of the image.

    Raises
    ------
    ValueError
        If the image array is not 2D (e.g. an RGB image with a channel
        axis; convert it to grayscale first).
    """
    if np.asarray(image_array).ndim != 2:
        raise ValueError(
            f"the image array must be 2D, got {np.asarray(image_array).ndim} "
            "dimensions"
        )

    if image_array.shape[0] > image_array.shape[1]:
        diff = image_array.shape[0] - image_array.shape[1]
        image_array = np.hstack((image_array, np.zeros((image_array.shape[0], diff))))
    elif image_array.shape[0] < image_array.shape[1]:
        diff = image_array.shape[1] - image_array.shape[0]
        image_array = np.vstack((image_array, np.zeros((diff, image_array.shape[1]))))

    base = 2
    order_y = 2

    while image_array.shape[0] > base:
        base = 2**order_y
        order_y += 1

    diffy = base - image_array.shape[0]

    if diffy != 0:
        image_array = np.vstack((image_array, np.zeros((diffy, image_array.shape[1]))))

    base = 2
    order_x = 2

    while image_array.shape[1] > base:
        base = 2**order_x
        order_x += 1

    diffx = base - image_array.shape[1]

    if diffx != 0:
        image_array = np.hstack((image_array, np.zeros((image_array.shape[0], diffx))))

    return image_array
