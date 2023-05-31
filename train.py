
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
from model import HeartDeformNet
from loss import *
from call_backs import *
from vtk_utils.vtk_utils import *
import yaml
import SimpleITK as sitk
"""# Set up"""

parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()
with open(args.config, 'r') as stream:
    params = yaml.safe_load(stream)

print('Finished parsing...')

img_shape = (params['network']['input_size'][0], params['network']['input_size'][1], params['network']['input_size'][2], 1)
save_model_path = os.path.join(params['train']['output_folder'], "weights_gcn.hdf5")

""" Create new directories """
try:
    os.makedirs(os.path.dirname(save_model_path))
    os.makedirs(os.path.dirname(params['train']['output_folder']))
except Exception as e: print(e)


"""# Feed in mesh info"""

pkl = pickle.load(open(params['train']['mesh_dat_filemame'], 'rb'))
mesh_info = construct_feed_dict(pkl, params['network']['num_blocks'], params['network']['coord_emb_dim'], has_cap=params['train']['loss']['if_cap'])

"""# Build the model"""
model = HeartDeformNet(params['train']['batch_size'], img_shape, params['network']['hidden_dim'], mesh_info, amplify_factor=params['network']['rescale_factor'],num_mesh=len(params['train']['data']['mesh_ids']), num_seg=params['network']['num_seg_class'], num_block=params['network']['num_blocks'],train=True)
unet_gcn = model.build_bc()

if params['train']['pre_train_unet'] is not None:
    unet_gcn = model.load_pre_trained_weights(unet_gcn, params['train']['pre_train_unet'],trainable=False)
unet_gcn.summary(line_length=150)

adam = Adam(lr=params['train']['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=True)
output_keys = [node.name.split('/')[0] for node in unet_gcn.outputs]
print("Output Keys: ", output_keys)
ctrl_loss_list = []
for i in range(params['network']['num_blocks']):
    ctrl_loss_list.append(ctrl_pts_loss_0()) 

losses = ctrl_loss_list
for i in range(params['network']['num_blocks']):
    losses += [mesh_loss_geometric_cf(mesh_info, params['train']['loss']['geom_wt'], k, params['train']['loss']['chamfer_ratio'], params['train']['loss']['mesh_wt'], params['train']['loss']['if_mask'], params['train']['loss']['turn_off_l2'], params['train']['loss']['if_cap']) for k in params['train']['data']['mesh_ids']]
if params['network']['num_seg_class'] >0:
    losses = [binary_bce_dice_loss] + losses 
print("OUTPUT_keys, losses: ", len(output_keys), len(losses))
losses = dict(zip(output_keys, losses))

metric_loss = []
metric_key = []
for i in range(1, len(params['train']['data']['mesh_ids'])+1):
    metric_key.append(output_keys[-i])
    metric_loss.append(mesh_point_loss_cf(params['train']['loss']['chamfer_ratio'], mesh_info['struct_node_ids'][i]-mesh_info['struct_node_ids'][i-1], params['train']['loss']['if_mask']))
    #metric_loss.append(mesh_point_loss_cf(params['train']['loss']['chamfer_ratio'], mesh_info['sample_node_ids'][i]-mesh_info['sample_node_ids'][i-1], params['train']['loss']['if_mask']))
print(metric_key, metric_loss)
metrics_losses = dict(zip(metric_key, metric_loss))
metric_loss_weights = list(np.ones(len(params['train']['data']['mesh_ids'])))
loss_weights = list(np.ones(len(output_keys)))
loss_weights[-len(params['train']['data']['mesh_ids']):] = [2.]*len(params['train']['data']['mesh_ids'])
if params['network']['num_seg_class'] > 0:
    loss_weights[0] = params['train']['loss']['seg_wt']
print("Current loss weights: ", loss_weights)


unet_gcn.compile(optimizer=adam, loss=losses,loss_weights=loss_weights,  metrics=metrics_losses)

""" Setup model checkpoint """
cp_cd = SaveModelOnCD(metric_key, save_model_path, patience=50)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.000005)
call_backs = [cp_cd, lr_schedule]

try:
    if params['train']['pre_train'] != '':
        unet_gcn.load_weights(params['train']['pre_train'])
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
if_seg = True if params['network']['num_seg_class']>0 else False

val_preprocessing_fn = functools.partial(_augment, **val_cfg)
train_ds_list, val_ds_list = [], []
train_ds_num, val_ds_num = [], []
for data_folder_out, attr in zip(params['train']['data']['train_img_folder'], params['train']['data']['train_sub_folder_attr']):
    x_train_filenames_i = buildImageDataset(data_folder_out, params['train']['data']['modality'], params['train']['data']['seed'], mode='_train'+attr, ext=params['train']['data']['file_pattern'])
    train_ds_num.append(len(x_train_filenames_i))
    train_ds_i = get_baseline_dataset(x_train_filenames_i, preproc_fn=tr_preprocessing_fn, mesh_ids=params['train']['data']['mesh_ids'], \
            shuffle_buffer=10000, if_seg=if_seg, num_block=params['network']['num_blocks'])
    train_ds_list.append(train_ds_i)
for data_val_folder_out, attr in zip(params['train']['data']['val_img_folder'], params['train']['data']['val_sub_folder_attr']):
    x_val_filenames_i = buildImageDataset(data_val_folder_out, params['train']['data']['modality'], params['train']['data']['seed'], mode='_val'+attr, ext=params['train']['data']['file_pattern'])
    val_ds_num.append(len(x_val_filenames_i))
    val_ds_i = get_baseline_dataset(x_val_filenames_i, preproc_fn=val_preprocessing_fn, mesh_ids=params['train']['data']['mesh_ids'], \
            shuffle_buffer=10000, if_seg=if_seg, num_block=params['network']['num_blocks'])
    val_ds_list.append(val_ds_i)
print("HELLO: ",params['train']['data']['train_sub_folder_attr'])
train_data_weights = [w/np.sum(params['train']['data']['train_sub_folder_weights']) for w in params['train']['data']['train_sub_folder_weights']]
val_data_weights = [w/np.sum(params['train']['data']['val_sub_folder_weights']) for w in params['train']['data']['val_sub_folder_weights']]
print("Sampling probability for train and val datasets: ", train_data_weights, val_data_weights)
train_ds = tf.data.experimental.sample_from_datasets(train_ds_list, weights=train_data_weights)
train_ds = train_ds.batch(params['train']['batch_size'])
val_ds = tf.data.experimental.sample_from_datasets(val_ds_list, weights=val_data_weights)
val_ds = val_ds.batch(params['train']['batch_size'])

num_train_examples = 1500
num_val_examples =  val_ds_num[np.argmax(val_data_weights)]/np.max(val_data_weights) 
print("Number of train, val samples after reweighting: ", num_train_examples, num_val_examples)

""" Training """
history =unet_gcn.fit(train_ds, 
                   steps_per_epoch=int(np.ceil(num_train_examples / float(params['train']['batch_size']))),
                   epochs=params['train']['num_epoch'],
                   validation_data=val_ds,
                   validation_steps= int(np.ceil(num_val_examples / float(params['train']['batch_size']))),
                   callbacks=call_backs)
with open(params['train']['output_folder']+"_history", 'wb') as handle: # saving the history 
        pickle.dump(history.history, handle)
