from numpy import asarray, max
from PIL import Image

from qtreemesh import QTree, QTreeMesh, image_preprocess

im = Image.open("5.jpg").convert('L')  # Converting to GrayScale
imar = image_preprocess(asarray(im))  # Creating an Array from Image
quad = QTree(None, imar, 70)
mesh = QTreeMesh(quad)
mesh.create_elements()

mesh.draw(True, 'orangered')

fem_nodes, fem_elements, fem_properties = mesh.adjust_mesh_for_FEM(True)

def triangle_area(x1, y1, x2, y2, x3, y3):
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

for i, ele in enumerate(fem_elements):
    if triangle_area(*fem_nodes[ele[0]-1],*fem_nodes[ele[1]-1],*fem_nodes[ele[2]-1]) == 0:
        print(*fem_nodes[ele[0]-1],*fem_nodes[ele[1]-1],*fem_nodes[ele[2]-1])

with open("nodes.txt", "w") as f:
    for i, (x, y) in enumerate(fem_nodes):
        xb = 0
        yb = 0
        if y < 0.001:
            xb, yb = -1, -1
        f.write(f"{i} {x} {y} {xb} {yb}\n")
        
with open("eles.txt", "w") as f:
    for i, ele in enumerate(fem_elements):
        t = 1
        if len(ele)==3:
            t = 3
        eles = [e-1 for e in ele]
        f.write(f"{i} {t} 0 " + " ".join(map(str, eles)) + "\n")
        
with open("mater.txt", "w") as f:
    for i, prop in enumerate(fem_properties):
        f.write(f"{1000 * (prop + 0.001)} 0.3\n")
        
fem_loads = []
top = max(fem_nodes[:,1])
for i,row in enumerate(fem_nodes):
    if row[1]>(top - 0.01):
        fem_loads.append([i, 0.0, 1.0])

with open("loads.txt", "w") as f:
    for load in fem_loads:
        f.write(" ".join(map(str, load)) + "\n")
        
from solidspy import solids_GUI
UC = solids_GUI()


from outputwrites import create_vtk_file
create_vtk_file(fem_nodes, fem_elements, UC)
