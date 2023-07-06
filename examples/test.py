import numpy as np
from PIL import Image

from qtreemesh import image_preprocess, QTree, QTreeMesh

im = Image.open("4.jpg").convert('L')  # Converting to GrayScale
imar = image_preprocess(np.asarray(im))  # Creating an Array from Image
quad = QTree(None, imar, 125)
mesh = QTreeMesh(quad)
mesh.create_elements()
mesh.draw(True, 'orangered')