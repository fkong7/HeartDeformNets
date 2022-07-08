import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../external"))
from vtk_utils.vtk_utils import *
import argparse

def face_intersection(fn):
    tmp_dir = os.path.join(os.path.dirname(fn), 'temp')
    try:
        os.makedirs(tmp_dir)
    except:
        pass

    mesh = load_vtk_mesh(fn)
    mesh = get_all_connected_polydata(mesh)
    write_vtk_polydata(mesh, 'debug.vtp')
    try:
        region_ids = np.unique(vtk_to_numpy(mesh.GetCellData().GetArray('Scalars_'))).astype(int)
    except:
        region_ids = np.unique(vtk_to_numpy(mesh.GetPointData().GetArray('RegionId'))).astype(int)
    total_num = 0
    bad_num = 0
    for i in region_ids:
        poly_i = thresholdPolyData(mesh, 'Scalars_', (i, i),'cell')
        if poly_i.GetNumberOfPoints() == 0:
            poly_i = thresholdPolyData(mesh, 'RegionId', (i, i),'point')
        name_i = os.path.basename(fn).split('.')[0] + '_{}'.format(i)
        out_fn_i = os.path.join(tmp_dir, name_i + '.stl')
        write_vtk_polydata(poly_i, out_fn_i)
        os.system('/Users/fanweikong/Downloads/tetgen1.6.0/tetgen -d ' + out_fn_i)
        if os.path.isfile(os.path.join(tmp_dir, name_i+'_skipped.face')):
            with open(os.path.join(tmp_dir, name_i+'_skipped.face')) as f:
                line = f.readline()
                bad_num += int(line.split(' ')[0])
        total_num += poly_i.GetNumberOfCells()
    print("Total_num, bad_num: ", total_num, bad_num)
    return bad_num/total_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn',  help='File name of the mesh.')
    args = parser.parse_args()
    percentage = face_intersection(args.fn) * 100.
    print("Percentage face intersection: ", percentage)
