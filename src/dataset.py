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
    tf.VarLenFeature = tf.io.VarLenFeature
    tf.FixedLenFeature = tf.io.FixedLenFeature
    tf.parse_single_example = tf.io.parse_single_example
    tf.sparse_tensor_to_dense = tf.sparse.to_dense
import functools
from augmentation import _augment
import numpy as np
def _parse_function_all(mode):
    def __parse(example_proto):
        if mode=='img':
            features = {"X": tf.VarLenFeature(tf.float32),
              "shape0": tf.FixedLenFeature((), tf.int64),
              "shape1": tf.FixedLenFeature((), tf.int64),
              "shape2": tf.FixedLenFeature((), tf.int64),
              }
            parsed_features = tf.parse_single_example(example_proto, features)
            img = tf.sparse_tensor_to_dense(parsed_features["X"])
            depth = tf.cast(parsed_features["shape0"], tf.int32)
            height = tf.cast(parsed_features["shape1"], tf.int32)
            width = tf.cast(parsed_features["shape2"], tf.int32)
            img = tf.reshape(img, tf.stack([depth,height, width, 1]))
            return img
        elif mode=='seg':
            features = {"S": tf.VarLenFeature(tf.int64),
              "shape0": tf.FixedLenFeature((), tf.int64),
              "shape1": tf.FixedLenFeature((), tf.int64),
              "shape2": tf.FixedLenFeature((), tf.int64),
              }
            parsed_features = tf.parse_single_example(example_proto, features)
            seg = tf.sparse_tensor_to_dense(parsed_features["S"])
            depth = tf.cast(parsed_features["shape0"], tf.int32)
            height = tf.cast(parsed_features["shape1"], tf.int32)
            width = tf.cast(parsed_features["shape2"], tf.int32)
            seg = tf.reshape(seg, tf.stack([depth,height, width, 1]))
            return seg
        elif 'mesh' in mode:
            mesh_id = mode.split('_')[-1]
            features = {"Y_"+mesh_id: tf.VarLenFeature(tf.float32)
              }
            parsed_features = tf.parse_single_example(example_proto, features)
            mesh = tf.sparse_tensor_to_dense(parsed_features["Y_"+mesh_id])

            #node_num = mesh.get_shape().as_list()[0] // 6
            node_num = tf.cast(tf.shape(mesh)[0]/6, tf.int32)
            mesh = tf.reshape(mesh, tf.stack([node_num, 6 ]))
            #slice_num = 8192
            #n_repeat = (node_num // slice_num)+1
            #mesh = tf.tile(mesh, [n_repeat, 1])
            #ids = tf.range(0, mesh.get_shape().as_list()[0])
            #sample_indices = tf.random.shuffle(ids)[:slice_num]
            #mesh = tf.gather(mesh, sample_indices, axis=0)
            #print("MESH SHAPE SAMPLED: ", n_repeat, mesh.get_shape().as_list())
            return mesh
        elif mode=='transform':
            features = {"Transform": tf.VarLenFeature(tf.float32)}
            parsed_features = tf.parse_single_example(example_proto, features)
            transform = tf.sparse_tensor_to_dense(parsed_features["Transform"])
            transform = tf.reshape(transform, [4, 4])
            return transform
        elif mode=='spacing':
            features = {"Spacing": tf.VarLenFeature(tf.float32)}
            parsed_features = tf.parse_single_example(example_proto, features)
            spacing = tf.sparse_tensor_to_dense(parsed_features["Spacing"])
            spacing = tf.reshape(spacing, [3])
            return spacing
        elif mode=='center':
            features = {"center": tf.VarLenFeature(tf.float32)}
            parsed_features = tf.parse_single_example(example_proto, features)
            center = tf.sparse_tensor_to_dense(parsed_features["center"])
            center = tf.reshape(center, [6])
            return center 
        elif mode=='grid_pts':
            return tf.zeros((1,1))
        else:
            raise ValueError('invalid name')

    return __parse


def get_baseline_dataset(filenames, preproc_fn=functools.partial(_augment),
                         threads=1, 
                         batch_size=0,
                         mesh_ids = [2], # default is LV blood pool 2
                         shuffle=True,
                         if_seg=True,
                         num_block=1,
                         shuffle_buffer=10000,
                         if_warp_im=False):           
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    files = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
        files = files.shuffle(shuffle_buffer)
    #dataset = files.apply(tf.contrib.data.parallel_interleave(
    #  tf.data.TFRecordDataset, cycle_length=threads))
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x),
                          cycle_length=threads, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset_img = dataset.map(_parse_function_all('img'))
    #dataset_trans = dataset.map(_parse_function_all('transform'))
    #dataset_spacing = dataset.map(_parse_function_all('spacing'))
    #dataset_center = dataset.map(_parse_function_all('center'))
    mesh_list = []
    for i in mesh_ids:
        dataset_mesh = dataset.map(_parse_function_all('mesh_'+str(i)))
        mesh_list.append(dataset_mesh)
    out_list = []
    grid_list = []
    im_list = []
    # Important node: we need to go through _parse_function_all instead of dataset from tensors tf.zeros((1,1))
    # otherwise it will generate the same dataset_center for all iterations
    for i in range(num_block):
        out_list += mesh_list
        dataset_grid = dataset.map(_parse_function_all('grid_pts'))
        grid_list.append(dataset_grid)
        if if_warp_im:
            dataset_im_diff = dataset.map(_parse_function_all('grid_pts'))
            im_list.append(dataset_im_diff)
    
    out_list = im_list + grid_list + out_list
    if if_seg:
        dataset_seg = dataset.map(_parse_function_all('seg'))
        out_list = [dataset_seg] + out_list

    #dataset_input = tf.data.Dataset.zip((dataset_img, dataset_trans, dataset_spacing))
    dataset_input = dataset_img
    dataset_output = tf.data.Dataset.zip(tuple(out_list))
    
    dataset = tf.data.Dataset.zip((dataset_input, dataset_output))
    dataset = dataset.map(preproc_fn)
    # It's necessary to repeat our data for all epochs
    dataset = dataset.repeat()
    if batch_size >0:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

