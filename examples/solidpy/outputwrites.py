import numpy as np
import pandas as pd

def create_vtk_file(coord,sdConn,d,CAU=None, SEC=None, Stress=None,filename = "output3.vtk"):
    f = open(filename, "w")
    f.write("# vtk DataFile Version 2.0\nOutput Data\nASCII\n")
    f.write("DATASET UNSTRUCTURED_GRID\n")
    numpoints = coord.shape[0]
    f.write("POINTS %i float\n"%numpoints)
    for each in coord:
        f.write("%f %f 0.0\n"%(each[0],each[1]))
    numcells = len(sdConn)
    newsdConn1 = [np.reshape(each,(1,-1))-1 for each in sdConn]
    newsdConn2 = [i[0,np.sort(np.unique(i,return_index=True)[1])] for i in newsdConn1]
    totdat = sum([i.shape[0] for i in newsdConn2])+len(newsdConn2)
    f.write("CELLS %i %i\n"%(numcells,totdat))
    for each in newsdConn2:
                
        f.write("%i "%each.shape[0])
        f.writelines(str(np.flip(each))[1:-1])
        f.write("\n")
    
    f.write("CELL_TYPES %i\n"%numcells)
    for i in range(numcells):
        f.write("7\n")
    
    f.write("POINT_DATA %i\n"%numpoints)
    f.write("SCALARS displacements float 3 \nLOOKUP_TABLE default \n")
    for row in d.reshape((-1,2)):
        f.write("%f %f 0.0\n"%(row[0],row[1]))
        
    if [Stress, CAU, SEC] != [None, None, None]:
        numcells = len(sdConn)
        f.write("CELL_DATA %i\n"%numcells)
    
    if Stress:
        f.write("SCALARS PK1 float 4 \nLOOKUP_TABLE default \n")
        for each in Stress:
            ms = np.mean(each, axis=1)
            f.write("%f %f %f %f\n"%(ms[0],ms[1],ms[2],ms[3]))
    
    if CAU:
        f.write("SCALARS Cauchy float 3 \nLOOKUP_TABLE default \n")
        for each in CAU:
            ms = np.mean(each, axis=1)
            f.write("%f %f %f\n"%(ms[0],ms[1],ms[2]))
            
    if SEC:
        f.write("SCALARS PK2 float 3 \nLOOKUP_TABLE default \n")
        for each in SEC:
            ms = np.mean(each, axis=1)
            f.write("%f %f %f\n"%(ms[0],ms[1],ms[2]))
    
    f.close()

