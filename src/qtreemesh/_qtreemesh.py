"""
A module for generating quadtree mesh from an image.

Author : Sadjad Abedi
"""

import numpy as np
from matplotlib.pyplot import figure, fill, show, axis


class Point():
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
        '''
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

        '''
        new_point = Point((self.x_coord + second_coord[0],
                           self.y_coord + second_coord[1]))

        return new_point


class QTree():
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
        The criteria used for partitioning. Default value is 1.
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
    def __init__(self, parent, array, crit=1, scale=1.,
                 bottom_left_corner=Point((0.0, 0.0)), depth = 0):
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

        self.property = np.mean(array)  # To define material properties by Averaging
        self.bottom_left_corner = bottom_left_corner  # BottomLeft Coordinates
        self.top_right_corner = bottom_left_corner.coord_sum(
            (array.shape[1]*scale, array.shape[0]*scale))  # TopRight Coordinates
        self.dimension = np.sqrt(array.size)  # To define scale requirement

        # SPLITTING
        if (np.max(array)-np.min(array)) > crit:  # Check Splitting Criteria
            self.sectors()

    def sectors(self):
        """
        A recursive function to create subtrees.

        Parameters
        ----------
        
        Returns
        -------
        None.

        """
        self.divided = True
        self.depth += 1
        size = self.array.shape
        bottom_left_north_west = self.bottom_left_corner.\
                                 coord_sum((0, (size[0]/2)*self.scale))
        self.north_west = QTree(self,self.array[0:size[0]//2, 0:size[1]//2],
                                self.crit, self.scale, bottom_left_north_west,
                                self.depth)
        bottom_left_north_east = self.bottom_left_corner.\
                                 coord_sum(((size[1]/2)*self.scale,
                                            (size[0]/2)*self.scale))
        self.north_east = QTree(self,self.array[0:size[0]//2, size[1]//2:size[1]],
                                self.crit, self.scale, bottom_left_north_east,
                                self.depth)
        bottom_left_south_west = self.bottom_left_corner
        self.south_west = QTree(self, self.array[size[0]//2:size[0], 0:size[1]//2],
                                self.crit, self.scale, bottom_left_south_west,
                                self.depth)
        bottom_left_south_east = self.bottom_left_corner.\
                                 coord_sum(((size[1]/2)*self.scale, 0))
        self.south_east = QTree(self,self.array[size[0]//2:size[0], size[1]//2:size[1]],
                                self.crit, self.scale, bottom_left_south_east,
                                self.depth)
    @property
    def count_leaves(self):
        """
        A recursive function to calculate the number of tree leaves.

        Parameters
        ----------
        node : QTree class
            A node of QTree.

        Returns
        -------
            Summation of leaves of subtrees

        """
        # If the node itself is "None" there are no leaves ->  return 0
        if self is None:
            return 0

        # If the node is a leaf then children will be "None" -> return 1
        if (self.north_west is None and self.north_east is None and
            self.south_west is None and self.south_east is None):
            return 1

        # Now we count the leaves in subtrees and return the sum
        return self.north_west.count_leaves +\
               self.north_east.count_leaves +\
               self.south_west.count_leaves +\
               self.south_east.count_leaves

    def save_leaves(self):
        """
        A function that stores all the leaves.
        
        Parameters
        ----------
        root : QTree class
            Root of the tree.

        Returns
        -------
        leaves_list : list
            A list of all leaves.    
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
                if (node.north_neighbor().south_west.divided or
                    node.north_neighbor().south_east.divided):
                    return True
        if node.south_neighbor() is not None:
            if node.south_neighbor().divided:
                if (node.south_neighbor().north_west.divided or
                    node.south_neighbor().north_east.divided):
                    return True
        if node.west_neighbor() is not None:
            if node.west_neighbor().divided:
                if (node.west_neighbor().north_east.divided or
                    node.west_neighbor().south_east.divided):
                    return True
        if node.east_neighbor() is not None:
            if node.east_neighbor().divided:
                if (node.east_neighbor().north_west.divided or
                    node.east_neighbor().south_west.divided):
                    return True

        return False

    def balancing(self):
        """
        Balance QTree for 2:1 ratio.

        Returns
        ------- 
        """
        leaves = self.save_leaves()
        while len(leaves) != 0:

            node = leaves.pop()
            if not node.divided:
                if self.need_split(node):
                    node.sectors()
                    leaves.extend([node.south_west,node.south_east,node.north_west,node.north_east])
                    if self.need_split(node.north_neighbor()):
                        leaves.append(node.north_neighbor())
                    if self.need_split(node.south_neighbor()):
                        leaves.append(node.south_neighbor())
                    if self.need_split(node.west_neighbor()):
                        leaves.append(node.west_neighbor())
                    if self.need_split(node.east_neighbor()):
                        leaves.append(node.east_neighbor())

class QTreeElement():
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
    


    Methods
    -------
    

    """
    def __init__(self,label,nodes_numbers,nodes_coordinates,
                 element_type,element_property) -> None:
        self.number = label
        self.nodes_numbers = nodes_numbers
        self.nodes_coordinates = nodes_coordinates
        self.element_type = element_type
        self.element_property = element_property

class QTreeMesh():
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
    


    Methods
    -------
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
    
    """
    def __init__(self, quad_tree:QTree, balancing = True) -> None:
        self.quad_tree = quad_tree
        if balancing:
            self.quad_tree.balancing()
        self.leaves = self.quad_tree.save_leaves()

        self.elements = []
        self.nodes = None

    def create_elements(self):
        """
        The main function of class that generate elements from quad-tree cells.
        """
        self.labeling()
        self.refactor_edge()
        for leaf in self.leaves:
            label = leaf.cell_number
            node_number = leaf.edge_points_numbers
            node_coordinate = [self.nodes[n-1,:] for n in node_number]
            element_type = leaf.cell_type
            element_property = leaf.property
            self.elements.append(QTreeElement(label,node_number,node_coordinate,
                                              element_type,element_property))



    def labeling(self):
        """
        A function that labels all cells and their corresponding corner
        points and add corner points to mesh nodes.   
        """
        self.nodes = np.array([[+np.inf,-np.inf]])
        label = 1

        for leaf in self.leaves:
            leaf.edge_points_numbers = []
            leaf.nodes_coordinate = []
            leaf.nodes_coordinate.append(np.array(leaf.bottom_left_corner.xy_coord))
            leaf.nodes_coordinate.append(np.array((leaf.top_right_corner.x_coord,
                            leaf.bottom_left_corner.y_coord)))
            leaf.nodes_coordinate.append(np.array(leaf.top_right_corner.xy_coord))

            leaf.nodes_coordinate.append(np.array((leaf.bottom_left_corner.x_coord,
                                      leaf.top_right_corner.y_coord)))
            for node in leaf.nodes_coordinate:  #Creating Element node list
                if any(np.equal(self.nodes,node).all(1)):
                    leaf.edge_points_numbers.append(
                        np.where(np.equal(self.nodes,node).all(1) == True)[0][0])
                else:
                    self.nodes = np.r_[self.nodes,[node]]
                    leaf.edge_points_numbers.append(self.nodes.shape[0]-1)

            leaf.cell_number = label
            label += 1
        self.nodes = self.nodes[1:,:]


    def refactor_edge(self):
        """
        A function that consider edge points, add them to 
        cells attributes, and detect cell modes based on the
        presence and location of edge points 
        """

        for leaf in self.leaves:
            newedge = list()
            mode = list()

            newedge.append(leaf.edge_points_numbers[0])
            if leaf.south_neighbor() is not None:
                if leaf.south_neighbor().divided:
                    newedge.append(leaf.south_neighbor().north_west.edge_points_numbers[3])
                    mode.append(True)
                else:
                    mode.append(False)
            else:
                mode.append(False)

            newedge.append(leaf.edge_points_numbers[1])
            if leaf.east_neighbor() is not None:
                if leaf.east_neighbor().divided:
                    newedge.append(leaf.east_neighbor().north_west.edge_points_numbers[0])
                    mode.append(True)
                else:
                    mode.append(False)
            else:
                mode.append(False)

            newedge.append(leaf.edge_points_numbers[2])
            if leaf.north_neighbor() is not None:
                if leaf.north_neighbor().divided:
                    newedge.append(leaf.north_neighbor().south_east.edge_points_numbers[0])
                    mode.append(True)
                else:
                    mode.append(False)
            else:
                mode.append(False)

            newedge.append(leaf.edge_points_numbers[3])
            if leaf.west_neighbor() is not None:
                if leaf.west_neighbor().divided:
                    newedge.append(leaf.west_neighbor().south_east.edge_points_numbers[2])
                    mode.append(True)
                else:
                    mode.append(False)
            else:
                mode.append(False)

            cell_type = self.mode_detection(mode)
            cell_type.append(leaf.dimension)
            leaf.edge_points_numbers = newedge
            leaf.cell_type = cell_type

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
            return [1,0]
        elif number_edge_points == 1:
            return [2,mode.index(True)*90]
        elif number_edge_points == 2:
            if mode == [True,True,False,False]:
                return [3,0]
            elif mode == [False,True,True,False]:
                return [3,90]
            elif mode == [False,False,True,True]:
                return [3,180]
            elif mode == [True,False,False,True]:
                return [3,270]
            elif mode == [True,False,True,False]:
                return [4,0]
            elif mode == [False,True,False,True]:
                return [4,90]
        elif number_edge_points == 3:
            return [5,mode.index(False)*90]
        elif number_edge_points == 4:
            return [6,0]

    def draw(self, fill_inside = True, edge_color = None, save_name = None):
        """
        Draw elements with filling inside.

        Parameters
        ----------
        fill_inside : bool, optional
            Fill elements with grayscale color based on 
            element property.
        edge_color : None/str, optional
            The color for element edges.
        save_name : None/str, optional
            name of file to save figure.

        Returns
        -------
            None.

        """
        fig = figure(figsize=(10,10),frameon=False)
        axis('off')
        for element in self.elements:
            fill([p[0] for p in element.nodes_coordinates],
                [p[1] for p in element.nodes_coordinates],
                facecolor = str(element.element_property/255) if fill_inside else 'white',
                edgecolor=edge_color)
        fig.tight_layout()
        show()
        if save_name:
            fig.savefig(save_name)


def image_preprocess(image_array):
    """
    A function to make image square and of order 2^n.

    Parameters
    ----------
    image_array : numpy array
        The array of the image.

    Returns
    -------
    image_array : numpy array
        The modified array of the image.
    """

    if image_array.shape[0] > image_array.shape[1]:
        diff = image_array.shape[0] - image_array.shape[1]
        image_array = np.hstack((image_array,np.zeros((image_array.shape[0],diff))))
    elif image_array.shape[0] < image_array.shape[1]:
        diff = image_array.shape[1] - image_array.shape[0]
        image_array = np.vstack((image_array,np.zeros((diff,image_array.shape[1]))))

    base = 2
    order_y = 2

    while image_array.shape[0] > base:
        base = 2**order_y
        order_y += 1

    diffy = base - image_array.shape[0]

    if diffy != 0:
        image_array = np.vstack((image_array,np.zeros((diffy,image_array.shape[1]))))

    base = 2
    order_x = 2

    while image_array.shape[1] > base:
        base = 2**order_x
        order_x += 1

    diffx = base - image_array.shape[1]

    if diffx != 0:
        image_array = np.hstack((image_array,np.zeros((image_array.shape[0],diffx))))

    return image_array
