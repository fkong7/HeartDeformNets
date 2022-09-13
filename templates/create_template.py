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
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
from utils import natural_sort, dice_score, write_scores
from vtk_utils.vtk_utils import *
from utils import *
import argparse 
import SimpleITK as sitk
from pre_process import resample_spacing
import glob

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_fn', default='', help='Name of the segmentation file')
    parser.add_argument('--if_turn_off_erode', action='store_true', default=False, help='If to turn off eroding the segmentation for better separation')
    parser.add_argument('--mesh_fn', default='', help='Name of the mesh file')
    parser.add_argument('--target_node_num', default=1000000, type=int, help='Number of node per class')
    parser.add_argument('--binary', action='store_true', help='Whether to convert to binary segmentation')
    parser.add_argument('--ref_im', default=None, help='Might need to resample the segmentation that came out of Slicer')
    parser.add_argument('--output', help='Filename to write to')
    return parser.parse_args()

def resample_segmentation(im, seg):
    # Without this the segmentation spacing will not be correct as Slicer changed it.
    new_segmentation = sitk.Resample(seg, im.GetSize(),
                                 sitk.Transform(),
                                 sitk.sitkNearestNeighbor,
                                 im.GetOrigin(),
                                 im.GetSpacing(),
                                 im.GetDirection(),
                                 0,
                                 seg.GetPixelID())
    sitk.WriteImage(new_segmentation, 'segmentation/wh.nii.gz')
    return new_segmentation

def convert_to_binary(seg, erode=True):
    left_id = [1, 2, 3, 6]
    right_id = [4, 5, 7]
    filt = sitk.BinaryMedianImageFilter()
    filt.SetRadius(3)
    # assemble chambers
    arr = sitk.GetArrayFromImage(seg)
    arr_total = np.zeros_like(arr) 
    for i in [1, 2, 3, 4, 5]:
        arr_total[arr==i] = 1
    total = filt.Execute(sitk.GetImageFromArray(arr_total))
    # erode a bit
    if erode:
        print("DEBUG: ERODING ...")
        ef = sitk.BinaryErodeImageFilter()
        ef.SetKernelRadius(3)
        ef.SetBackgroundValue(0)
        ef.SetForegroundValue(1)
        total = ef.Execute(total)
    arr_total = sitk.GetArrayFromImage(total)
    
    # assemble Ao-LV, PA-RV
    arr_aolv = np.zeros_like(arr)
    arr_parv = np.zeros_like(arr)
    for i in [3, 6]:
        arr_aolv[arr==i] = 1
    for i in [5, 7]:
        arr_parv[arr==i] = 1
    aolv = filt.Execute(sitk.GetImageFromArray(arr_aolv))
    parv = filt.Execute(sitk.GetImageFromArray(arr_parv))
    arr_aolv = sitk.GetArrayFromImage(aolv)
    arr_parv = sitk.GetArrayFromImage(parv)
    
    arr_total[arr_aolv>0] = 1
    arr_total[arr_parv>0] = 1
    
    # mask the multilabel segmentation
    arr[arr_total==0] = 0
    seg_b = sitk.GetImageFromArray(arr_total)
    seg_b.CopyInformation(seg)
    seg_m = sitk.GetImageFromArray(arr)
    seg_m.CopyInformation(seg)
    return seg_b, seg_m

def create_tmplt(seg, target_num):
    template = convert_to_surfs(seg, new_spacing=[0.3, 0.3, 0.3], target_node_num=target_num)
    template = smooth_polydata(template, 25)
    return create_tmplt_mesh(template, seg)

def create_tmplt_mesh(mesh, ref):
    SIZE = (128, 128, 128)
    img_center = np.array(ref.TransformContinuousIndexToPhysicalPoint(np.array(ref.GetSize())/2.0))
    ref_r = resample_spacing(ref, template_size=SIZE, order=1)[0]  # numpy array
    img_center2 = np.array(ref_r.TransformContinuousIndexToPhysicalPoint(np.array(ref_r.GetSize())/2.0))
    transform = build_transform_matrix(ref_r)
    mesh  = transform_polydata(mesh, img_center2-img_center, transform, SIZE)
    return mesh, ref_r[0]
    

if __name__ == '__main__':
    args = parse()
    if args.seg_fn != '':
        seg = sitk.ReadImage(args.seg_fn)
        if args.binary:
            seg_b, seg_m = convert_to_binary(seg, not args.if_turn_off_erode)
            if args.ref_im is not None:
                seg_b = resample_segmentation(sitk.ReadImage(args.ref_im), seg_b)
                seg_m = resample_segmentation(sitk.ReadImage(args.ref_im), seg_m)
            tmplt_manifold, _ = create_tmplt(seg_b, args.target_node_num*8)
            write_vtk_polydata(tmplt_manifold, args.output[:-4]+'_manifold.vtp')
            write_vtk_polydata(tmplt_manifold, args.output[:-4]+'_manifold.obj')
            tmplt, _ = create_tmplt(seg_m, args.target_node_num)
        else:
            if args.ref_im is not None:
                seg = resample_segmentation(sitk.ReadImage(args.ref_im), seg)
            tmplt, seg_r = create_tmplt(seg, args.target_node_num)
            sitk.WriteImage(seg_r, args.output[:-3]+'nii.gz')
    else:
        tmplt, _ = create_tmplt_mesh(load_vtk_mesh(args.mesh_fn), sitk.ReadImage(args.ref_im))
    write_vtk_polydata(tmplt, args.output)
    write_vtk_polydata(tmplt, args.output[:-3]+'vtp')

