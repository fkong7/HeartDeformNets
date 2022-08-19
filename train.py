
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
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "external"))
import glob
import functools
import pickle
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
print("TENSORFLOW VERSION: ", tf.__version__)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras import models

from utils import buildImageDataset, construct_feed_dict 
from custom_layers import *

from augmentation import changeIntensity_img, _augment
from dataset import get_baseline_dataset
from model import HeartDeepFFD
from loss import *
from call_backs import *
from vtk_utils.vtk_utils import *

import SimpleITK as sitk
"""# Set up"""

parser = argparse.ArgumentParser()
parser.add_argument('--im_trains', nargs='+', help='Name of the folder containing the image data')
parser.add_argument('--im_vals', nargs='+', help='Name of the folder containing the image data')
parser.add_argument('--file_pattern', default='*.tfrecords', help='Pattern of the .tfrecords files')
parser.add_argument('--pre_train_im', default='', help="Filename of the pretrained unet")
parser.add_argument('--pre_train', default='', help="Filename of the pretrained model")
parser.add_argument('--mesh',  help='Name of the .dat file containing mesh info')
parser.add_argument('--attr_trains', nargs='+', help='Attribute name of the folders containing tf records')
parser.add_argument('--attr_vals', nargs='+', help='Attribute name of the folders containing tf records')
parser.add_argument('--train_data_weights', type=float, nargs='+', help='Weights to apply for the samples in different datasets')
parser.add_argument('--val_data_weights', type=float, nargs='+', help='Weights to apply for the samples in different datasets')
parser.add_argument('--output',  help='Name of the output folder')
parser.add_argument('--modality', nargs='+', help='Name of the modality, mr, ct, split by space')
parser.add_argument('--num_epoch', type=int, help='Maximum number of epochs to run')
parser.add_argument('--coord_emb_dim', default=192, type=int, help='Dimension of embeded vertex coordinates')
parser.add_argument('--num_seg', type=int,default=1, help='Number of segmentation classes')
parser.add_argument('--num_block', type=int,default=3, help='Number of graph conv block')
parser.add_argument('--seg_weight', type=float, default=1., help='Weight of the segmentation loss')
parser.add_argument('--geom_weights', type=float, default=[0.5, 0.5], nargs='+', help='Weight of the geometric accuracy loss')
parser.add_argument('--mesh_weights', type=float, default=[0.5, 0.5, 0.5], nargs='+', help='Weight of the S, V, and S-V L2 loss')
parser.add_argument('--mesh_ids', nargs='+', type=int, default=[2], help='Number of meshes to train')
parser.add_argument('--seed', type=int, default=41, help='Randome seed')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
parser.add_argument('--shuffle_buffer_size', type=int, default=10000, help='Shuffle buffer size')
parser.add_argument('--lr', type=float, help='Learning rate')
parser.add_argument('--cf_ratio', type=float, default=1., help='Loss ratio between gt chamfer loss and pred chamfer loss')
parser.add_argument('--size', type = int, nargs='+', help='Image dimensions')
parser.add_argument('--hidden_dim', type = int, default=128, help='Hidden dimension')
parser.add_argument('--amplify_factor', type=float, default=0.1, help="amplify_factor of the predicted displacements")
parser.add_argument('--if_mask', action='store_true', help='If to mask out out of bound predictions in point loss function.')
parser.add_argument('--if_cap', action='store_true', help='If to apply regularization losses on the vessel caps.')
parser.add_argument('--turn_off_l2', action='store_false', help='Turn off the L2 difference between S and V.')
parser.add_argument('--cp_loss', help='Use L2 of laplace coord or Euclidean difference from the centroid for displacement regularization')

args = parser.parse_args()
print('Finished parsing...')

img_shape = (args.size[0], args.size[1], args.size[2], 1)
save_model_path = os.path.join(args.output, "weights_gcn.hdf5")

""" Create new directories """
try:
    os.makedirs(os.path.dirname(save_model_path))
    os.makedirs(os.path.dirname(args.output))
except Exception as e: print(e)


"""# Feed in mesh info"""

pkl = pickle.load(open(args.mesh, 'rb'))
mesh_info = construct_feed_dict(pkl, args.num_block, args.coord_emb_dim, has_cap=args.if_cap)

"""# Build the model"""
model = HeartDeepFFD(args.batch_size, img_shape, args.hidden_dim, mesh_info, amplify_factor=args.amplify_factor,num_mesh=len(args.mesh_ids), num_seg=args.num_seg, num_block=args.num_block,train=True)
unet_gcn = model.build_bc()

if args.pre_train_im != '':
    unet_gcn = model.load_pre_trained_weights(unet_gcn, args.pre_train_im,trainable=False)
unet_gcn.summary(line_length=150)

adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=True)
output_keys = [node.name.split('/')[0] for node in unet_gcn.outputs]
print("Output Keys: ", output_keys)
ctrl_loss_list = []
for i in range(args.num_block):
    ctrl_loss_list.append(ctrl_pts_loss_0()) 

losses = ctrl_loss_list
for i in range(args.num_block):
    losses += [mesh_loss_geometric_cf(mesh_info, args.geom_weights, k, args.cf_ratio, args.mesh_weights, args.if_mask, args.turn_off_l2, args.if_cap) for k in args.mesh_ids]
if args.num_seg >0:
    losses = [binary_bce_dice_loss] + losses 
print("OUTPUT_keys, losses: ", len(output_keys), len(losses))
losses = dict(zip(output_keys, losses))

metric_loss = []
metric_key = []
for i in range(1, len(args.mesh_ids)+1):
    metric_key.append(output_keys[-i])
    metric_loss.append(mesh_point_loss_cf(args.cf_ratio, mesh_info['struct_node_ids'][i]-mesh_info['struct_node_ids'][i-1], args.if_mask))
    #metric_loss.append(mesh_point_loss_cf(args.cf_ratio, mesh_info['sample_node_ids'][i]-mesh_info['sample_node_ids'][i-1], args.if_mask))
print(metric_key, metric_loss)
metrics_losses = dict(zip(metric_key, metric_loss))
metric_loss_weights = list(np.ones(len(args.mesh_ids)))
loss_weights = list(np.ones(len(output_keys)))
loss_weights[-len(args.mesh_ids):] = [2.]*len(args.mesh_ids)
if args.num_seg > 0:
    loss_weights[0] = args.seg_weight
print("Current loss weights: ", loss_weights)


unet_gcn.compile(optimizer=adam, loss=losses,loss_weights=loss_weights,  metrics=metrics_losses)

""" Setup model checkpoint """
save_model_path = os.path.join(args.output, "weights_gcn.hdf5")

cp_cd = SaveModelOnCD(metric_key, save_model_path, patience=50)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.000005)
weight_schedule = ReduceLossWeight(grid_weight, patience=5, factor=0.95)
call_backs = [cp_cd, lr_schedule, weight_schedule]

try:
    if args.pre_train != '':
        unet_gcn.load_weights(args.pre_train)
    else:
        print("Loading model, ", save_model_path)
        unet_gcn.load_weights(save_model_path)
except Exception as e:
  print("Model not loaded", e)
  pass

"""## Set up train and validation datasets
Note that we apply image augmentation to our training dataset but not our validation dataset.
"""
tr_cfg = {'changeIntensity': {"scale": [0.9, 1.1],"shift": [-0.1, 0.1]}}
tr_preprocessing_fn = functools.partial(_augment, **tr_cfg)
val_cfg = {}
if_seg = True if args.num_seg>0 else False

val_preprocessing_fn = functools.partial(_augment, **val_cfg)
train_ds_list, val_ds_list = [], []
train_ds_num, val_ds_num = [], []
for data_folder_out, attr in zip(args.im_trains, args.attr_trains):
    x_train_filenames_i = buildImageDataset(data_folder_out, args.modality, args.seed, mode='_train'+attr, ext=args.file_pattern)
    train_ds_num.append(len(x_train_filenames_i))
    train_ds_i = get_baseline_dataset(x_train_filenames_i, preproc_fn=tr_preprocessing_fn, mesh_ids=args.mesh_ids, \
            shuffle_buffer=args.shuffle_buffer_size, if_seg=if_seg, num_block=args.num_block)
    train_ds_list.append(train_ds_i)
for data_val_folder_out, attr in zip(args.im_vals, args.attr_vals):
    x_val_filenames_i = buildImageDataset(data_val_folder_out, args.modality, args.seed, mode='_val'+attr, ext=args.file_pattern)
    val_ds_num.append(len(x_val_filenames_i))
    val_ds_i = get_baseline_dataset(x_val_filenames_i, preproc_fn=val_preprocessing_fn, mesh_ids=args.mesh_ids, \
            shuffle_buffer=args.shuffle_buffer_size, if_seg=if_seg, num_block=args.num_block)
    val_ds_list.append(val_ds_i)
train_data_weights = [w/np.sum(args.train_data_weights) for w in args.train_data_weights]
val_data_weights = [w/np.sum(args.val_data_weights) for w in args.val_data_weights]
print("Sampling probability for train and val datasets: ", train_data_weights, val_data_weights)
train_ds = tf.data.experimental.sample_from_datasets(train_ds_list, weights=train_data_weights)
train_ds = train_ds.batch(args.batch_size)
val_ds = tf.data.experimental.sample_from_datasets(val_ds_list, weights=val_data_weights)
val_ds = val_ds.batch(args.batch_size)

num_train_examples = 1500
num_val_examples =  val_ds_num[np.argmax(val_data_weights)]/np.max(val_data_weights) 
print("Number of train, val samples after reweighting: ", num_train_examples, num_val_examples)

""" Training """
history =unet_gcn.fit(train_ds, 
                   steps_per_epoch=int(np.ceil(num_train_examples / float(args.batch_size))),
                   epochs=args.num_epoch,
                   validation_data=val_ds,
                   validation_steps= int(np.ceil(num_val_examples / float(args.batch_size))),
                   callbacks=call_backs)
with open(args.output+"_history", 'wb') as handle: # saving the history 
        pickle.dump(history.history, handle)
