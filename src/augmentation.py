
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
from packaging import version
import tensorflow as tf
TF2 = version.parse(tf.__version__) >= version.parse('2.0')
if TF2:
    tf.random_uniform = tf.random.uniform

def shift_img(output_imVg, label_img, width_shift_range, height_shift_range):
  return output_img, label_img

"""## Flipping the image randomly"""
def flip_img(horizontal_flip, tr_img, label_img):
  return tr_img, label_img

"""## Rotate the image with random angle""" 
def rotate_img(rotation, tr_img, label_img):
    return tr_img, label_img

"""##Scale/shift the image intensity randomly"""

def changeIntensity_img(tr_img,  change):
  if change:
    scale = tf.random_uniform([], change['scale'][0], change['scale'][1])
    shift = tf.random_uniform([], change['shift'][0], change['shift'][1])
    tr_img = tr_img*scale+shift
    tr_img = tf.clip_by_value(tr_img, -1., 1.)
  return tr_img

def mask_img(tr_img, mean_size):
    l = tf.random.normal([], mean=mean_size, stddev=mean_size//3)
    m = tf.random.normal([], mean=mean_size, stddev=mean_size//3)
    n = tf.random.normal([], mean=mean_size, stddev=mean_size//3)
    s1 = tf.cast(tf.shape(tr_img)[0], dtype=tf.float32)
    s2 = tf.cast(tf.shape(tr_img)[1], dtype=tf.float32)
    s3 = tf.cast(tf.shape(tr_img)[2], dtype=tf.float32)
    x = tf.random.uniform([], 0, s1//3)
    y = tf.random.uniform([], 0, s2//3) 
    z = tf.random.uniform([], 0, s3//3)
    x_n = tf.clip_by_value(x+l, 0, s1)
    y_n = tf.clip_by_value(y+m, 0, s2)
    z_n = tf.clip_by_value(z+n, 0, s3)
    grid = tf.meshgrid(tf.linspace(0., s1-1., tf.shape(tr_img)[0]), 
            tf.linspace(0., s2-1., tf.shape(tr_img)[1]), 
            tf.linspace(0., s3-1., tf.shape(tr_img)[2]))
    mask = tf.logical_and(tf.logical_and(grid[0]>=x, grid[0]<x_n), tf.logical_and(grid[1]>=y, grid[1]<y_n))
    mask = tf.logical_and(mask, tf.logical_and(grid[2]>=z, grid[2]<z_n))
    rand_img = tf.random.uniform(tf.shape(tr_img), -1., 1.)
    new_img = tf.where(tf.expand_dims(mask, -1), rand_img, tr_img)
    return new_img

def resample_img(tr_img, size):
    ori_size = tf.shape(tr_img)[1]
    ori_size_float = tf.cast(ori_size, tf.float32)
    size = tf.cast(tf.clip_by_value(tf.random.normal([], mean=size, stddev=size//3), 5., ori_size_float), tf.uint16)
    axis = tf.cast(tf.random.uniform([], 0, 3), dtype=tf.int32) # axis to resize
    tr_img = tf.squeeze(tr_img, -1)
    axis_list = tf.convert_to_tensor([0, 1, 2], dtype=tf.int32)
    axis_list = tf.roll(axis_list, shift=-axis, axis=0)
    tr_img = tf.transpose(tr_img, axis_list)
    tr_img = tf.image.resize(tr_img, [size, ori_size], method='nearest')

    axis_list_back = tf.convert_to_tensor([0, 1, 2], dtype=tf.int32)
    axis_list_back = tf.roll(axis_list_back, shift=axis, axis=0)
    tr_img =  tf.image.resize(tr_img, [ori_size, ori_size], method='bilinear')
    tr_img = tf.transpose(tr_img, axis_list_back)
    tr_img = tf.expand_dims(tr_img, -1)
    return tr_img

"""## Assembling our transformations into our augment function"""
def _augment(inputs, outputs,
             changeIntensity=False,
             size=128,
             mask_prob=1,
             mask_mean_size=0):
  inputs = changeIntensity_img(inputs, changeIntensity)
  inputs = tf.cond(tf.random.uniform([])>mask_prob, lambda: resample_img(inputs, size), lambda: inputs)
  #inputs = tf.cond(tf.random.uniform([])>mask_prob, lambda: mask_img(inputs, mask_mean_size), lambda: inputs)
  return inputs, outputs
