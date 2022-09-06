'''
This script uses VTK's CleanPolyData filter to merge points that are adjacent together.
When applied on whole heart meshes generated from marching cube, the adjacent cardiact
structures can thus be merged together.
This will create a connected graph for the whole heart for farthest sampling of the 
control handles
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../external'))
from vtk_utils.vtk_utils import *
import argparse
import vtk

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', help='File name of the template mesh')
    parser.add_argument('--output', help='Output file name of the template mesh')
    parser.add_argument('--decimate_rate', default=0., type=float, help='Percentage of decimation')
    args = parser.parse_args()

    mesh = load_vtk_mesh(args.fn)
    get_range = mesh.GetBounds()
    extend = 1./3.*(get_range[1]-get_range[0] + \
            get_range[3] - get_range[2] + \
            get_range[5] - get_range[4])
    mesh = cleanPolyData(mesh, extend/50.)
    mesh = decimation(mesh, args.decimate_rate)
    write_vtk_polydata(mesh, args.output)

