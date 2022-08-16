#Copyright (C) 2022 Fanwei Kong, Shawn C. Shadden, University of California, Berkeley

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
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
        return ids

    def FindPointsWithinRadius(self, r, pt):
        ids = vtk.vtkIdList()
        self.locator.FindPointsWithinRadius(r, pt, ids)
        return ids

def get_next_ring_neighbors(mesh, point_list):
    next_ring_neighbors = set()
    for point_id in point_list:
        cell_ids = vtk.vtkIdList()
        mesh.GetPointCells(point_id, cell_ids)
        connected_pt_ids = vtk.vtkIdList()
        for j in range(cell_ids.GetNumberOfIds()):
            mesh.GetCellPoints(cell_ids.GetId(j), connected_pt_ids)
            for k in range(connected_pt_ids.GetNumberOfIds()):
                pt_id = connected_pt_ids.GetId(k)
                if pt_id not in point_list:
                    next_ring_neighbors.add(pt_id)
    return next_ring_neighbors

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

def deform_with_contact(mesh_init, mesh_end, out_dir):
    # get coordinates of the two meshes in numpy arrays
    coords_init = vtk_to_numpy(mesh_init.GetPoints().GetData())
    coords_end = vtk_to_numpy(mesh_end.GetPoints().GetData())
    displacements = coords_end - coords_init 
    # loop over time iterations
    num_time_steps = 100
    radius = 0.01
    rings = 3
    # 2. find n-ring neighbors (we can probably move this step outside of the loop and store this information for efficiency)
    neighbor_list = []
    locator_pts = vtk.vtkPoints()
    locator_pts.SetData(numpy_to_vtk(coords_init))
    locator = PointLocator(locator_pts)
    for p in range(coords_init.shape[0]):
        input_ids = set([p])
        for r in range(rings):
            next_ids = get_next_ring_neighbors(mesh_init, input_ids)
            input_ids = next_ids | input_ids
        r_ids_vtk = locator.FindPointsWithinRadius(radius*2, coords_init[p,:])
        r_ids = set([r_ids_vtk.GetId(z) for z in range(r_ids_vtk.GetNumberOfIds())])
        input_ids = r_ids | input_ids
        input_ids.remove(p)
        neighbor_list.append(input_ids)
    
    #print(neighbor_list[0])
    for i in range(num_time_steps):
        # update mesh coordinates
        coords_init += displacements * 1./float(num_time_steps)
        locator_pts = vtk.vtkPoints()
        locator_pts.SetData(numpy_to_vtk(coords_init))
        locator = PointLocator(locator_pts)
        rebuild = False
        # loop over each point
        for p in range(coords_init.shape[0]):
            n_ids = neighbor_list[p]
            # 1. find N closest points
            if rebuild:
                print("rebuilding locator")
                locator_pts = vtk.vtkPoints()
                locator_pts.SetData(numpy_to_vtk(coords_init))
                locator = PointLocator(locator_pts)
                rebuild = False
            r_ids = locator.FindPointsWithinRadius(radius, coords_init[p,:])
            #print(n_ids)
            #print([r_ids.GetId(z) for z in range(r_ids.GetNumberOfIds())])
            # 3. move points that are too close (in the direction of -dx?)
            correction = np.zeros(3)
            count = 0
            for q in range(r_ids.GetNumberOfIds()):
                close_id = r_ids.GetId(q)
                if (close_id in n_ids) or (close_id == p):
                    continue
                else:
                    count += 1
                    # compute distance 
                    vec = coords_init[close_id,:] - coords_init[p,:]
                    length = np.linalg.norm(vec)
                    if length < 2.*radius:
                        corr = (2.*radius - length) * vec / length
                        correction -= corr
            if count > 0:
                coords_init[p,:] += correction /float(count)/float(num_time_steps)
                rebuild = True
        # 4. smooth (might be better to smooth the mesh only locally
        #mesh_init = smooth_vtk_polydata(mesh_init) # this is a global smoothing
        write_vtk_polydata(mesh_init, os.path.join(out_dir, 'iter_{}.vtp'.format(i)))
    return mesh_init

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn_init',  help='File name of the template mesh.')
    parser.add_argument('--fn_end',  help='File name of the deformed mesh.')
    parser.add_argument('--fn_output',  help='File name of the deformed mesh.')
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.fn_output)):
        os.makedirs(os.path.dirname(args.fn_output))
    mesh_init = load_vtk_mesh(args.fn_init)
    mesh_end = load_vtk_mesh(args.fn_end)

    mesh_result = deform_with_contact(mesh_init, mesh_end, os.path.dirname(args.fn_output))

    mesh_result = write_vtk_polydata(mesh_result, args.fn_output)
