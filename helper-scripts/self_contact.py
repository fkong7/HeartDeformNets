import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../external"))
from vtk_utils.vtk_utils import *
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import argparse

class PointLocator:
    # Class to find closest points
    def __init__(self,pts):
        ds = vtk.vtkPolyData()
        ds.SetPoints(pts)
        self.locator = vtk.vtkKdTreePointLocator()
        self.locator.SetDataSet(ds)
        self.locator.BuildLocator()
    def findNClosestPoints(self,pt, N):
        ids = vtk.vtkIdList()
        self.locator.FindClosestNPoints(N, pt, ids)
        return

def get_one_ring_neighbors(mesh, point_id):
    one_ring_neighbors = set()
    
    cell_ids = vtk.vtkIdList()
    mesh.GetPointCells(point_id, cell_ids)
    connected_pt_ids = vtk.vtkIdList()
    for j in range(cell_ids.GetNumberOfIds()):
        mesh.GetCellPoints(cell_ids.GetId(j), connected_pt_ids)
        for k in range(connected_pt_ids.GetNumberOfIds()):
            pt_id = connected_pt_ids.GetId(k)
            if pt_id != point_id:
                one_ring_neighbors.add(pt_id)
    return one_ring_neighbors

def smooth_vtk_polydata(mesh, iteration=1, boundary=False, feature=False):

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(mesh)
    smoother.SetBoundarySmoothing(boundary)
    smoother.SetFeatureEdgeSmoothing(feature)
    smoother.SetNumberOfIterations(iteration)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    smoothed = smoother.GetOutput()

    return smoothed

def deform_with_contact(mesh_init, mesh_end):
    # get coordinates of the two meshes in numpy arrays
    coords_init = vtk_to_numpy(mesh_init.GetPoints().GetData())
    coords_end = vtk_to_numpy(mesh_end.GetPoints().GetData())
    displacements = coords_end - coords_init 

    # loop over time iterations
    num_time_steps = 10
    for i in range(num_time_steps):
        # update mesh coordinates
        coords_init += displacements * 1./float(num_time_steps)
        
        # loop over each point
        for p in range(coords_init.shape[0]):
            # 1. find N closest points
            # 2. find 1-ring neighbors (we can probably move this step outside of the loop and store this information for efficiency)
            n_ids = get_one_ring_neighbors(mesh_init,p)
            # 3. move points that are too close (in the direction of -dx?)
            # 4. smooth (might be better to smooth the mesh only locally
            #mesh_init = smooth_vtk_polydata(mesh_init) # this is a global smoothing

    return mesh_init

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn_init',  help='File name of the template mesh.')
    parser.add_argument('--fn_end',  help='File name of the deformed mesh.')
    parser.add_argument('--fn_output',  help='File name of the deformed mesh.')
    args = parser.parse_args()

    mesh_init = load_vtk_mesh(args.fn_init)
    mesh_end = load_vtk_mesh(args.fn_end)

    mesh_result = deform_with_contact(mesh_init, mesh_end)

    mesh_result = write_vtk_polydata(mesh_result, args.fn_output)
