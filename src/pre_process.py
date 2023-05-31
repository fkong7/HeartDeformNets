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
import numpy as np
import SimpleITK as sitk
from copy import deepcopy
  
def resample(sitkIm, resolution = (0.5, 0.5, 0.5),order=1,dim=3):
    if type(sitkIm) is str:
        image = sitk.ReadImage(sitkIm)
    else:
        image = sitkIm
    resample = sitk.ResampleImageFilter()
    if order==1:
        resample.SetInterpolator(sitk.sitkLinear)
    else:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputSpacing(resolution)

    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = np.array(image.GetSpacing())
    new_size = orig_size*(orig_spacing/np.array(resolution))
    new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    newimage = resample.Execute(image)
    
    return newimage

def cropMask(mask, percentage):
  ori_shape = mask.shape
  # crop the surroundings by percentage
  def boolCounter(boolArr):
      #Count consecutive occurences of values varying in length in a numpy array
      out = np.diff(np.where(np.concatenate(([boolArr[0]],
                                       boolArr[:-1] != boolArr[1:],
                                       [True])))[0])[::2]
      return out
  dim  = len(mask.shape)
  for i in range(dim):
      tmp = np.moveaxis(mask, i, 0)
      IDs = np.max(np.max(tmp,axis=-1),axis=-1)==0
      blank = boolCounter(IDs)
      upper = int(blank[0]*percentage) if int(blank[0]*percentage) != 0 else 1
      lower = -1*int(blank[-1]*percentage) if int(blank[-1]*percentage) !=0 else -1
      mask = np.moveaxis(tmp[int(blank[0]*percentage): -1*int(blank[-1]*percentage),:,:],0,i)
  ratio = np.array(mask.shape)/np.array(ori_shape)
  return mask, ratio


def transform_func(image, reference_image, transform, order=1):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    if order ==1:
        interpolator = sitk.sitkLinear
    elif order == 0:
        interpolator = sitk.sitkNearestNeighbor
    elif order ==3:
        interpolator = sitk.sitkBSpline
    default_value = 0
    try:
        resampled = sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)
    except Exception as e: print(e)

    return resampled

def reference_image_build(spacing, size, template_size, dim):
    #template size: image(array) dimension to resize to: a list of three elements
    reference_size = template_size
    reference_spacing = np.array(size)/np.array(template_size)*np.array(spacing)
    reference_spacing = np.mean(reference_spacing)*np.ones(3)
    #reference_size = size
    reference_image = sitk.Image(reference_size, 0)
    reference_image.SetOrigin(np.zeros(3))
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(np.eye(3).ravel())
    return reference_image

def centering(img, ref_img, order=1):
    dimension = img.GetDimension()
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - ref_img.GetOrigin())
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    reference_center = np.array(ref_img.TransformContinuousIndexToPhysicalPoint(np.array(ref_img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    if int(sitk.__version__.split('.')[0]) > 1:
        centered_transform = sitk.CompositeTransform([centered_transform, centering_transform])
    else:
        centered_transform.AddTransform(centering_transform)

    return transform_func(img, ref_img, centered_transform, order)

def isometric_transform(image, ref_img, orig_direction, order=1, target=None):
    # transform image volume to orientation of eye(dim)
    dim = ref_img.GetDimension()
    affine = sitk.AffineTransform(dim)
    if target is None:
        target = np.eye(dim)
    
    ori = np.reshape(orig_direction, np.eye(dim).shape)
    target = np.reshape(target, np.eye(dim).shape)
    affine.SetMatrix(np.matmul(target,np.linalg.inv(ori)).ravel())
    affine.SetCenter(ref_img.TransformContinuousIndexToPhysicalPoint(np.array(ref_img.GetSize())/2.0))
    #affine.SetMatrix(image.GetDirection())
    return transform_func(image, ref_img, affine, order)

def resample_spacing(sitkIm, resolution=0.5, dim=3, template_size=(256, 256, 256), order=1):
    if type(sitkIm) is str:
        image = sitk.ReadImage(sitkIm)
    else:
        image = sitkIm
    orig_direction = image.GetDirection()
    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = np.array(image.GetSpacing())
    new_size = orig_size*(orig_spacing/np.array(resolution))
    new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    new_size = np.abs(np.matmul(np.reshape(orig_direction, (3,3)), np.array(new_size)))
    ref_img = reference_image_build(resolution, new_size, template_size, dim)
    centered = centering(image, ref_img, order)
    transformed = isometric_transform(centered, ref_img, orig_direction, order)
    return transformed, ref_img

def resample_scale(sitkIm, ref_img, scale_factor=1., order=1):
    assert type(scale_factor)==np.float64, "Isotropic scaling"
    dim = sitkIm.GetDimension()
    affine = sitk.AffineTransform(dim)
    scale = np.eye(dim)
    np.fill_diagonal(scale, 1./scale_factor)
    
    affine.SetMatrix(scale.ravel())
    affine.SetCenter(sitkIm.TransformContinuousIndexToPhysicalPoint(np.array(sitkIm.GetSize())/2.0))
    transformed = transform_func(sitkIm, ref_img, affine, order)
    return transformed

class SpatialTransform(object):
    '''
    Base class to image transform
    '''
    def __init__(self, image, mask=None, mesh=None):
        self.image = image
        self.dim = image.GetDimension()
        self.mask = mask
        self.mesh = mesh
    def set_input(self, image, mask=None, mesh=None):
        self.image = image
        self.dim = image.GetDimension()
        if mask is not None:
            self.mask = mask
        if mesh is not None:
            self.mesh = mesh
    def clear_transform(self):
        self.transform = sitk.Transform()
    def apply_transform(self):
        output = []
        out_im = transform_func(self.image, self.image, self.transform, order=1)
        output.append(out_im)
        if self.mask is not None:
            out_mask = transform_func(self.mask, self.mask, self.transform, order=0)
            output.append(out_mask)
        if self.mesh is not None:
            #out_mesh = np.copy(self.mesh)
            #Had to do a copy like this not sure why
            out_mesh = np.zeros(self.mesh.shape)
            #inv = self.transform.GetInverse()
            for i in range(self.mesh.shape[0]):
                out_mesh[i,:] = self.mesh[i,:]
                out_mesh[i,:] = self.transform.TransformPoint(out_mesh[i,:])

            output.append(out_mesh)
        return output
    def add_transform(self, transform):
        total = sitk.Transform(self.transform)
        total.AddTransform(transform)
        self.transform = total

class AffineTransform(SpatialTransform):
    '''
    Apply random affine transform to input 3D image volume
    '''
    def __init__(self, image, shear_range, scale_range, rot_range, trans_range, flip_prob, mask=None, mesh=None):
        super(AffineTransform, self).__init__(image, mask, mesh)
        self.shear_range = shear_range
        self.scale_range = scale_range
        self.rot_range = rot_range
        self.flip_prob = flip_prob
        self.trans_range = trans_range
        self.transform = sitk.Transform()

    def scale(self):
        scale_trans= sitk.AffineTransform(self.dim)
        scale = np.eye(self.dim)
        scale = np.diag(np.random.uniform(self.scale_range[0], self.scale_range[1], self.dim))
        scale_trans.SetMatrix(scale.ravel())
        scale_trans.SetCenter(self.image.TransformContinuousIndexToPhysicalPoint(np.array(self.image.GetSize())/2.0))
        self.transform.AddTransform(scale_trans)

    def rotate(self):
        angles = np.random.uniform(self.rot_range[0], self.rot_range[1], self.dim)
        rads = np.array(angles)/180.*np.pi
        x_rot = np.eye(self.dim)
        x_rot = [[1., 0., 0.], [0., np.cos(rads[0]), -np.sin(rads[0])], [0., np.sin(rads[0]), np.cos(rads[0])]]
        y_rot = [[np.cos(rads[1]), 0., np.sin(rads[1])], [0.,1.,0.], [-np.sin(rads[1]), 0., np.cos(rads[1])]]
        z_rot = [[np.cos(rads[2]), -np.sin(rads[2]), 0.], [np.sin(rads[2]), np.cos(rads[2]), 0.], [0., 0., 1.]]
        rot_matrix = np.matmul(np.matmul(np.array(x_rot), np.array(y_rot)), np.array(z_rot))
        rotate_trans = sitk.AffineTransform(3)
        rotate_trans.SetMatrix(rot_matrix.ravel())
        rotate_trans.SetCenter(self.image.TransformContinuousIndexToPhysicalPoint(np.array(self.image.GetSize())/2.0))
        self.transform.AddTransform(rotate_trans)
    
    def translate(self):
        t_trans = sitk.AffineTransform(3)
        params = np.random.uniform(self.trans_range[0],self.trans_range[1], self.dim)
        t_trans.SetTranslation(params)
        self.transform.AddTransform(t_trans)

    def shear(self):
        shear_trans = sitk.AffineTransform(3)  
        axis = np.argsort(np.random.rand(self.dim))
        shear_trans.Shear(int(axis[0]), int(axis[1]), np.random.uniform(self.shear_range[0], 
            self.shear_range[1]))
        shear_trans.SetCenter(self.image.TransformContinuousIndexToPhysicalPoint(np.array(self.image.GetSize())/2.0))
        self.transform.AddTransform(shear_trans)
    
    def flip(self):
        flip = np.random.rand(self.dim)>self.flip_prob
        flip_matrix = np.eye(self.dim)
        flip_matrix[np.diag(flip)] = -1. 
        flip_trans = sitk.AffineTransform(3)
        flip_trans.SetMatrix(flip_matrix.ravel())
        flip_trans.SetCenter(self.image.TransformContinuousIndexToPhysicalPoint(np.array(self.image.GetSize())/2.0))
        self.transform.AddTransform(flip_trans)

    def affine(self):
        self.rotate()
        self.shear()
        self.scale()
        self.translate()
        #self.flip()
    def apply_transform(self):
        output = []
        out_im = transform_func(self.image, self.image, self.transform, order=1)
        output.append(out_im)
        if self.mask is not None:
            out_mask = transform_func(self.mask, self.mask, self.transform, order=0)
            output.append(out_mask)
        if self.mesh is not None:
            #out_mesh = np.copy(self.mesh)
            #Had to do a copy like this not sure why
            out_mesh = np.zeros(self.mesh.shape)
            #We have to use a inv transform on the points - it's SimpleITK's decision that
            #the resampling transform on image is defined from output to input
            inv = self.transform.GetInverse()
            for i in range(self.mesh.shape[0]):
                out_mesh[i,:] = self.mesh[i,:]
                out_mesh[i,:] = inv.TransformPoint(out_mesh[i,:])

            output.append(out_mesh)
        return output

class NonlinearTransform(SpatialTransform):
    '''
    b-spline transform 
    '''
    def __init__(self, image, num_ctrl_pts, stdev, mask=None, mesh=None):
        super(NonlinearTransform, self).__init__(image, mask, mesh)
        self.num_ctrl_pts = num_ctrl_pts
        self.stdev = stdev

    def bspline(self):
        from scipy.interpolate import RegularGridInterpolator
        transform_mesh_size = [self.num_ctrl_pts] * self.dim
        self.transform = sitk.BSplineTransformInitializer(
            self.image ,
            transform_mesh_size
        )
        d = np.random.randn(self.num_ctrl_pts+3, self.num_ctrl_pts+3, self.num_ctrl_pts+3,3)*self.stdev
        d[:2, :, :, :] = 0.
        d[-2:, :, :, :] = 0.
        d[:, :2, :, :] = 0.
        d[:, -2:, :, :] = 0.
        d[:, :, :2, :] = 0.
        d[:, :, -2:, :] = 0.
        sitk.WriteImage(sitk.GetImageFromArray(d[:,:,:,0]), '/Users/fanweikong/Documents/Modeling/HeartDeepFFD/output/debug/b_spline.nii.gz')
        
        params = np.asarray(self.transform.GetParameters(), dtype=np.float64)
        params += d.flatten(order='F')
        self.transform.SetParameters(tuple(params))
    def apply_transform(self):
        from scipy.optimize import minimize
        output = []
        out_im = transform_func(self.image, self.image, self.transform, order=1)
        output.append(out_im)
        if self.mask is not None:
            out_mask = transform_func(self.mask, self.mask, self.transform, order=0)
            output.append(out_mask)
        if self.mesh is not None:
            #out_mesh = np.copy(self.mesh)
            #Had to do a copy like this not sure why
            out_mesh = np.zeros(self.mesh.shape)
            #We have to use a inv transform on the points - it's SimpleITK's decision that
            #the resampling transform on image is defined from output to input
            for i in range(self.mesh.shape[0]):
                out_mesh[i,:] = self.mesh[i,:]
                def fun(x):
                    return np.linalg.norm(self.transform.TransformPoint(x) - out_mesh[i,:])
                p = np.array(out_mesh[i,:])
                res = minimize(fun, p, method='Powell')
                out_mesh[i,:] = res.x
            output.append(out_mesh)
        return output


def swapLabels_ori(labels):
    labels[labels==421]=420
    unique_label = np.unique(labels)

    new_label = range(len(unique_label))
    for i in range(len(unique_label)):
        label = unique_label[i]
        newl = new_label[i]
        labels[labels==label] = newl
    

    return labels

def swapLabels(labels):
    labels[labels==421]=420
    unique_label = np.unique(labels)

    new_label = range(len(unique_label))
    for i in range(len(unique_label)):
        label = unique_label[i]
        newl = new_label[i]
        labels[labels==label] = newl
    
    if len(unique_label) != 4:
        labels[labels==1] = 0
        labels[labels==4] = 0
        labels[labels==5] = 0
        labels[labels==7] = 0
        labels[labels==2] = 1
        labels[labels==3] = 2
        labels[labels==6] = 3
       

    return labels
  
def swapLabelsBack(labels,pred):
    labels[labels==421]=420
    unique_label = np.unique(labels)
    new_label = range(len(unique_label))

    for i in range(len(unique_label)):
      pred[pred==i] = unique_label[i]
      
    return pred
    

def RescaleIntensity(slice_im,m,limit, bound=[-1, 1]):
    if type(slice_im) != np.ndarray:
        raise RuntimeError("Input image is not numpy array")
    #slice_im: numpy array
    #m: modality, ct or mr
    if m =="ct":
        rng = abs(limit[0]-limit[1])
        threshold = rng/2
        slice_im[slice_im>limit[0]] = limit[0]
        slice_im[slice_im<limit[1]] = limit[1]
        #(slice_im-threshold-np.min(slice_im))/threshold
        slice_im = slice_im/threshold
        slice_im -= np.min(slice_im)
        slice_im  = slice_im/2*(bound[1] - bound[0])
        slice_im += bound[0]
    elif m=="mr":
        #slice_im[slice_im>limit[0]*2] = limit[0]*2
        #rng = np.max(slice_im) - np.min(slice_im)
        pls = np.unique(slice_im)
        #upper = np.percentile(pls, 99)
        #lower = np.percentile(pls, 10)
        upper = np.percentile(slice_im, 99)
        lower = np.percentile(slice_im, 20)
        slice_im[slice_im>upper] = upper
        slice_im[slice_im<lower] = lower
        slice_im -= int(lower)
        rng = upper - lower
        slice_im = slice_im/rng*(bound[1] - bound[0])
        slice_im += bound[0]
    return slice_im


    
