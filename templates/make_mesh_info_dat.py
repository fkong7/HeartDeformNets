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
sys.path.append(os.path.join(os.path.dirname(__file__), '../external'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import networkx as nx
import numpy as np
import argparse 
from datetime import datetime
import scipy.sparse as sp
import pickle
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy import sparse
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from vtk_utils.vtk_utils import *
from pre_process import *
import trimesh
import yaml

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))
    
    return sparse_to_tuple(t_k)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def unit(vec):
    return np.array(vec)/np.linalg.norm(vec)

def find_cap(node_id, mesh, tag_id=1, max_cap_num=1000000, tol = 0.5, side_ring_num=4):
    print("Finding cap for node {} and assigning tag id {}...".format(node_id, tag_id))
    try:
        vtk_arr = mesh.GetCellData().GetArray('Face_ID')
        face_data = vtk_to_numpy(vtk_arr)
        vtk_arr_s = mesh.GetCellData().GetArray('Side_ID')
        side_data = vtk_to_numpy(vtk_arr_s)
    except:
        face_data = np.zeros(mesh.GetNumberOfCells())
        side_data = np.zeros(mesh.GetNumberOfCells())
    ctr_cells, side_cells, face_cells = [], [], []

    if node_id > -1:
        pt_norms = get_point_normals(mesh)
        ctr_nrm = pt_norms[node_id,:]
        queue = [node_id]
        visited = []
        count = 0
        while len(queue) >0 and count < max_cap_num:
            count += 1
            c_arr = vtk.vtkIdList()
            n = queue.pop(0)
            visited.append(n)
            mesh.GetPointCells(n, c_arr)
            if len(ctr_cells) == 0: #record cells connected to the center
                ctr_cells = [c_arr.GetId(ci) for ci in range(c_arr.GetNumberOfIds())]
            for c_id in range(c_arr.GetNumberOfIds()):
                p_arr = vtk.vtkIdList()
                mesh.GetCellPoints(c_arr.GetId(c_id), p_arr)
                found = True
                for p_id in range(p_arr.GetNumberOfIds()):
                    PID = p_arr.GetId(p_id)
                    if np.sum(unit(pt_norms[PID, :]) * unit(ctr_nrm))>tol:
                        if PID not in visited:
                            visited.append(PID)
                            queue.append(PID)
                    else:
                        found = False
                        if c_id not in side_cells:
                            side_cells.append(c_arr.GetId(c_id))
                        break
                if found:
                    face_cells.append(c_arr.GetId(c_id))
            #print("DEBUG: count: ", count, face_cells)
    face_data[face_cells] = tag_id
    # find rings of cells on the side
    constraint_set = face_cells
    add_cells = [q for q in face_cells]
    for _ in range(side_ring_num):
        add_cells, constraint_set = find_connected_points(add_cells, mesh, constraint_set)
        side_cells += add_cells
    side_data[side_cells] = tag_id
    vtk_arr = numpy_to_vtk(face_data)
    vtk_arr.SetName('Face_ID')
    mesh.GetCellData().AddArray(vtk_arr)
    vtk_arr_s = numpy_to_vtk(side_data)
    vtk_arr_s.SetName('Side_ID')
    mesh.GetCellData().AddArray(vtk_arr_s)
    return mesh, ctr_cells, side_cells, face_data

   
def cal_lap_index(mesh_neighbor):
    print("Mesh neighbor: ", mesh_neighbor.shape)
    max_len = 0
    for j in mesh_neighbor:
        if len(j) > max_len:
            max_len = len(j)
    print("Max vertex degree is ", max_len)
    lap_index = np.zeros([mesh_neighbor.shape[0], max_len+2]).astype(np.int32)
    for i, j in enumerate(mesh_neighbor):
        lenj = len(j)
        lap_index[i][0:lenj] = j
        lap_index[i][lenj:-2] = -1
        lap_index[i][-2] = i
        lap_index[i][-1] = lenj
    return lap_index

def get_face_node_list(mesh, output_mesh=False, target_num=None, cap_list=None):
    mesh_info = {'node_list': [0], 'face_list': [], 'lap_list': [], 'cap_id_list':[], 'face_side_list':[], 'face_ctr_list':[]}
    total_node = 0
    region_ids = np.unique(vtk_to_numpy(mesh.GetPointData().GetArray('RegionId'))).astype(int)
    if output_mesh:
        poly = []
    for index, i in enumerate(region_ids):
        poly_i = thresholdPolyData(mesh, 'RegionId', (i, i),'point')
        if target_num is not None:
            rate = max(0., 1. - float(target_num)/poly_i.GetNumberOfPoints())    
            poly_i = decimation(poly_i, rate)
        num_pts = poly_i.GetNumberOfPoints()
        total_node += poly_i.GetNumberOfPoints()
        mesh_info['node_list'].append(total_node)
        cells = vtk_to_numpy(poly_i.GetPolys().GetData()) 
        cells = cells.reshape(poly_i.GetNumberOfCells(), 4)
        #print(cells)
        cells = cells[:,1:]
        # calculate neighbors using trimesh
        tri = trimesh.Trimesh(vertices=vtk_to_numpy(poly_i.GetPoints().GetData()), faces=(cells), process=False)
        mesh_info['lap_list'].append(cal_lap_index(np.array(tri.vertex_neighbors)))
        mesh_info['face_list'].append(cells)
        if cap_list is not None:
            ctr_list = []
            side_list = []
            for c_i, c in enumerate(cap_list[index]):
                poly_i, ctr_i, side_i, cap_id_i = find_cap(c, poly_i, tag_id=c_i+1)
                ctr_list.append(ctr_i)
                side_list.append(side_i)
            mesh_info['cap_id_list'].append(cap_id_i)
            mesh_info['face_side_list'].append(side_list)
            mesh_info['face_ctr_list'].append(ctr_list)
        if output_mesh:
            poly.append(poly_i)
    if output_mesh:
        vtk_poly = appendPolyData(poly)
        return mesh_info, vtk_poly
    else:
        return mesh_info 

def get_center_ids(mesh, coords_fn, num_mesh):
    if coords_fn is None:
        return [[-1]]*num_mesh
    else:
        id_list = []
        with open(coords_fn, 'r') as f:
            params = yaml.safe_load(f)
            for k in params.keys():
                if params[k] is None:
                    id_list.append([-1])
                else:
                    mesh_k = thresholdPolyData(mesh, 'RegionId', (k, k),'point')
                    locator = vtk.vtkKdTreePointLocator()
                    locator.SetDataSet(mesh_k) 
                    locator.BuildLocator()
                    id_list.append([locator.FindClosestPoint(pt) for pt in params[k]])
        return id_list

def make_dat_bc(mesh_fn, sample_fn, weight_fns, ctrl_fns, write=True, output_path=None, num_mesh=7, coords_fn=None):
    template = load_vtk_mesh(mesh_fn)
    sample_mesh = load_vtk_mesh(sample_fn)
    ctrl_pts = [load_vtk_mesh(ctrl_fn) for ctrl_fn in ctrl_fns]
    for i, p in enumerate(ctrl_pts):
        write_polydata_points(p, ctrl_fns[i][:-3]+'vtp')
    # Break template mesh into different cardiac structures
    NUM_MESH = num_mesh
    # cap dict
    cap_list = get_center_ids(template, coords_fn, num_mesh)
    print("DEBUG: ", cap_list)
    tmplt_mesh_info, template = get_face_node_list(template, cap_list=cap_list, output_mesh=True)
    sample_mesh_info = get_face_node_list(sample_mesh)
    coords = vtk_to_numpy(template.GetPoints().GetData())
    weights = [np.genfromtxt(weight_fn,delimiter=',').astype(np.float32) for weight_fn in weight_fns]

    sample_pts = vtk_to_numpy(sample_mesh.GetPoints().GetData())

    # find the corresponding id of ctrl pts on sample pts
    id_ctrl_on_sample_all = []
    for j in range(len(weights)):
        # find the corresponding id of ctrl pts on sample pts
        ctrl_coords = vtk_to_numpy(ctrl_pts[j].GetPoints().GetData())
        id_ctrl_on_sample = find_point_correspondence(sample_mesh, ctrl_pts[j].GetPoints())
        sample_pts[id_ctrl_on_sample, :] = ctrl_coords
        id_ctrl_on_sample_all.append(id_ctrl_on_sample)
    
    #----------Not needed if sample mesh and template mesh are the same-----
    id_mesh_on_sample = []
    for i in range(NUM_MESH):
        sample_i = vtk.vtkPolyData()
        sample_i_pts = vtk.vtkPoints()
        sample_i_pts.SetData(numpy_to_vtk(sample_pts[sample_mesh_info['node_list'][i]: sample_mesh_info['node_list'][i+1], :]))
        sample_i.SetPoints(sample_i_pts)
        tmplt_i = vtk.vtkPolyData()
        tmplt_i_pts = vtk.vtkPoints()
        tmplt_i_pts.SetData(numpy_to_vtk(coords[tmplt_mesh_info['node_list'][i]:tmplt_mesh_info['node_list'][i+1],:]))
        tmplt_i.SetPoints(tmplt_i_pts)
        id_mesh_on_sample.append(find_point_correspondence(tmplt_i, sample_i_pts))
    #print(id_mesh_on_sample)
    #-----------------------------------------------------------------------
    info = {'sample_coords': sample_pts.astype(np.float32), 'sample_faces': sample_mesh_info['face_list'], 'sample_node_list': sample_mesh_info['node_list'], \
            'bbw': weights , 'tmplt_coords': coords.astype(np.float32), \
            'id_ctrl_on_sample_all': id_ctrl_on_sample_all, \
            'cap_data':tmplt_mesh_info['cap_id_list'], 'cap_ctr_data': tmplt_mesh_info['face_ctr_list'], 'cap_side_data': tmplt_mesh_info['face_side_list']}

    # build graph for contrl pts
    sample_faces_total = vtk_to_numpy(sample_mesh.GetPolys().GetData())
    sample_faces_total = sample_faces_total.reshape(sample_mesh.GetNumberOfCells(), 4)
    mesh = trimesh.Trimesh(vertices=sample_pts, faces=(sample_faces_total[:,1:]), process=False)
    info['lap_ids'] = cal_lap_index(mesh.vertex_neighbors)
    adj_1 = nx.adjacency_matrix(mesh.vertex_adjacency_graph, nodelist=range(len(sample_pts)))
    cheb_1 = chebyshev_polynomials(adj_1,1)
    info['support'] = cheb_1

    if write:
        t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        pickle.dump(info, open(os.path.join(output_path, t +"_bbw.dat"),"wb"), protocol=2)
        template.GetPointData().RemoveArray('BC')
        sample_mesh.GetPointData().RemoveArray('BC')
        write_vtk_polydata(template, os.path.join(output_path, t+"_template.vtp"))
        write_vtk_polydata(sample_mesh, os.path.join(output_path, t+"_sample.vtp"))
    return info

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmplt_fn', help='Name of the template mesh.')
    parser.add_argument('--sample_fn', help='Name of the sampling mesh.')
    parser.add_argument('--ctrl_fns', nargs='+', default=[], help='Name of the control points')
    parser.add_argument('--weight_fns', nargs='+', default=[], help='Name of the control points')
    parser.add_argument('--out_dir', help='Path to the output folder')
    parser.add_argument('--center_coords_fn', default=None, help='Filename where coordinates of the center points of vessels were stored for assigning regularization losses')
    parser.add_argument('--num_mesh', type=int, help='Number of mesh components')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    make_dat_bc(args.tmplt_fn, args.sample_fn, args.weight_fns, args.ctrl_fns, write=True, output_path=args.out_dir, num_mesh=args.num_mesh, coords_fn=args.center_coords_fn)

