from numpy import asarray
from PIL import Image

from qtreemesh import QTree, QTreeMesh, image_preprocess

im = Image.open("4.jpg").convert('L')  # Converting to GrayScale
imar = image_preprocess(asarray(im))  # Creating an Array from Image
quad = QTree(None, imar, 125)
mesh = QTreeMesh(quad)
mesh.create_elements()
mesh.draw(True, 'orangered')
