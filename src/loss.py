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
from packaging import version
import tensorflow as tf
TF2 = version.parse(tf.__version__) >= version.parse('2.0')
from tensorflow.keras import losses
from tensorflow.keras import backend as K
import numpy as np

from tensorflow.python.framework import ops
if TF2: 
    nn_distance_module=tf.load_op_library('/global/home/users/fanwei_kong/3DPixel2Mesh/external_2/tf_nndistance_so.so')
else:
    nn_distance_module=tf.load_op_library('/global/home/users/fanwei_kong/3DPixel2Mesh/external/tf_nndistance_so.so')

def nn_distance(xyz1,xyz2):
    '''
Computes the distance of nearest neighbors for a pair of point clouds
input: xyz1: (batch_size,#points_1,3)  the first point cloud
input: xyz2: (batch_size,#points_2,3)  the second point cloud
output: dist1: (batch_size,#point_1)   distance from first to second
output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second
output: dist2: (batch_size,#point_2)   distance from second to first
output: idx2:  (batch_size,#point_2)   nearest neighbor from second to first
    '''
    return nn_distance_module.nn_distance(xyz1,xyz2)
#@ops.RegisterShape('NnDistance')
#def _nn_distance_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(3)
#    shape2=op.inputs[1].get_shape().with_rank(3)
#    return [tf.TensorShape([shape1.dims[0],shape1.dims[1]]),tf.TensorShape([shape1.dims[0],shape1.dims[1]]),
#        tf.TensorShape([shape2.dims[0],shape2.dims[1]]),tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
@ops.RegisterGradient('NnDistance')
def _nn_distance_grad(op,grad_dist1,grad_idx1,grad_dist2,grad_idx2):
    xyz1=op.inputs[0]
    xyz2=op.inputs[1]
    idx1=op.outputs[1]
    idx2=op.outputs[3]
    return nn_distance_module.nn_distance_grad(xyz1,xyz2,grad_dist1,idx1,grad_dist2,idx2)

def unit(tensor):
    return tf.nn.l2_normalize(tensor, axis=-1)

def laplace_coord(pred, lap_ids):
    batch_size = tf.shape(pred)[0]
    # Add one zero vertex since the laplace index was initialized to be -1 
    vertex = tf.concat([pred, tf.zeros([batch_size, 1, 3])], 1)
    indices = lap_ids[:,:-2]
    weights = tf.cast(lap_ids[:,-1], tf.float32)
    weights = tf.tile(tf.reshape(tf.reciprocal(weights), [-1,1]), [1,3])
    laplace = tf.reduce_sum(tf.gather(vertex, indices, axis=1), 2)
    laplace = tf.subtract(pred, tf.multiply(laplace, weights))
    return laplace

def ctrl_pts_loss(weight, lap_ids):
    def ctrl_pts_loss_k(y_true, y_pred):
        lap = laplace_coord(y_pred, lap_ids)
        pt_loss = tf.reduce_mean(tf.reduce_sum(tf.square(lap), -1))
        return pt_loss * weight
    return ctrl_pts_loss_k

def ctrl_pts_loss_l1(weight):
    def ctrl_pts_loss_l1_k(y_true, y_pred):
        mean_pt = tf.reduce_mean(y_pred, axis=1, keepdims=True)
        diff = y_pred - mean_pt
        pt_loss = tf.reduce_mean(tf.abs(diff), -1)
        return pt_loss * weight
    return ctrl_pts_loss_l1_k

def ctrl_pts_loss_l2(weight):
    def ctrl_pts_loss_l2_k(y_true, y_pred):
        mean_pt = tf.reduce_mean(y_pred, axis=1, keepdims=True)
        diff = y_pred - mean_pt
        pt_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), -1))
        return pt_loss * weight
    return ctrl_pts_loss_l2_k

def ctrl_pts_loss_0(weight):
    def ctrl_pts_loss_0_k(y_true, y_pred):
        return tf.zeros((1,))
    return ctrl_pts_loss_0_k 

def mesh_loss_geometric_cf(feed_dict, weights, mesh_id, cf_ratio=1., if_mask=True, if_l2=True):
    def loss(y_true, y_pred):
        losses = mesh_loss(y_pred, y_true, feed_dict, mesh_id, cf_ratio, if_mask, if_l2)
        point_loss, normal_loss, edge_loss, lap_loss, cap_loss  = losses
        total_loss = tf.pow(point_loss*10, weights[0])*tf.pow(normal_loss*10, weights[1]) + 20.*cap_loss
        return total_loss
    return loss

def mesh_loss_geometric_cf_ffd(feed_dict, weights, mesh_id, cf_ratio=1., if_mask=True, if_l2=True):
    def loss(y_true, y_pred):
        ctr_list = feed_dict['ctr_data'][mesh_id]
        side_list = feed_dict['side_data'][mesh_id]
        cap_id_data = feed_dict['cap_data'][mesh_id]
        point_loss, normal_loss, edge_loss, lap_loss, cap_loss = mesh_loss_single(y_pred, y_true, feed_dict['tmplt_faces'][mesh_id], None, ctr_list, side_list, cap_id_data, cf_ratio, if_mask, mesh_id=mesh_id)
        total_loss = tf.pow(point_loss*10, weights[0])*tf.pow(normal_loss*10, weights[1]) + 20.*cap_loss
        return total_loss
    return loss

def mesh_point_loss_cf(cf_ratio=1., mesh_n=0, if_mask=True):
    def point_loss_cf(y_true, y_pred):
        gt_pt = y_true[:, :, :3]
        y_pred = y_pred[:, :mesh_n, :]
        dist1,idx1,dist2,idx2 = nn_distance(gt_pt, y_pred) # dist1: from gt to pred; dist2: from pred to gt
        if if_mask:
            dist2_bounded = tf.gather_nd(dist2, tf.where(tf.logical_and(tf.reduce_all(y_pred<tf.reduce_max(gt_pt, axis=1, keepdims=True),-1), tf.reduce_all(y_pred>tf.reduce_min(gt_pt, axis=1, keepdims=True), -1))), batch_dims=0)
            pred2gt_err = tf.where(tf.shape(dist2_bounded)[-1]>0 , tf.reduce_mean(dist2_bounded), tf.reduce_mean(dist1))
            point_loss = (2.-cf_ratio)*tf.reduce_mean(dist1) + cf_ratio* pred2gt_err
        else:
            point_loss = (2.-cf_ratio)*tf.reduce_mean(dist1) + cf_ratio*tf.reduce_mean(dist2)
        return point_loss
    return point_loss_cf

def mesh_loss_single(pred, gt, tmplt_faces, lap_ids, ctr_list, side_list, cap_id_data, cf_ratio=1., if_mask=False, mesh_id=0):
    #weights = [0.1, 0.6, 0.3]
    gt_pt = gt[:, :, :3] # gt points
    gt_nm = gt[:, :, 3:] # gt normals
    v1 = tf.gather(pred, tmplt_faces[:,0], axis=1) # B NF 3
    v2 = tf.gather(pred, tmplt_faces[:,1], axis=1)
    v3 = tf.gather(pred, tmplt_faces[:,2], axis=1)
    e1 = v2 - v1
    e2 = v3 - v1
    e3 = v3 - v2
    cross = unit(tf.linalg.cross(e1, e2))
    ctr = (v1 + v2 + v3)/3.
    cap_loss = calc_cap_normal_loss(cross, ctr_list, side_list, cap_id_data, mesh_id)
    e1_l = tf.reduce_sum(tf.square(e1), axis=-1)
    e2_l = tf.reduce_sum(tf.square(e2), axis=-1)
    e3_l = tf.reduce_sum(tf.square(e3), axis=-1)
    edge_loss_1 = tf.abs(e1_l - tf.reduce_mean(e1_l, axis=-1, keepdims=True))
    edge_loss_2 = tf.abs(e2_l - tf.reduce_mean(e2_l, axis=-1, keepdims=True))
    edge_loss_3 = tf.abs(e3_l - tf.reduce_mean(e3_l, axis=-1, keepdims=True))
    edge_loss = (edge_loss_1 + edge_loss_2 + edge_loss_3)/3.
    edge_loss = tf.reduce_mean(edge_loss)
    
    if lap_ids is not None:
        lap = laplace_coord(pred, lap_ids)
        lap_loss = tf.reduce_mean(tf.reduce_sum(tf.square(lap), -1))
    else:
        lap_loss = 0.

    dist_gt2pred, idx_gt2pred, dist_pred2gt, idx_pred2gt = nn_distance(gt_pt, ctr) 
    
    gather_ids = tf.expand_dims(idx_pred2gt, axis=-1) #[#batch, NF, 1]
    normal = unit(tf.gather_nd(gt_nm, gather_ids, batch_dims=1)) #[#batch, NF, 3]
    normal_dist = tf.reduce_sum(tf.square(normal-cross), axis=-1)  #[#batch, NF]

    gather_ids_2 = tf.expand_dims(idx_gt2pred, axis=-1)
    normal_2 = unit(tf.gather_nd(cross, gather_ids_2, batch_dims=1))
    normal_dist_2 = tf.reduce_sum(tf.square(normal_2 - gt_nm), axis=-1)

    extra_pt_err, extra_nrm_err = 0., 0.
    if if_mask:
        pred2gt_err = masked_loss(dist_pred2gt, ctr, gt_pt, tf.reduce_mean(dist_gt2pred))
        normal_loss = masked_loss(normal_dist, ctr, gt_pt, tf.reduce_mean(normal_dist))
    else:
        pred2gt_err = tf.reduce_mean(dist_pred2gt)
        normal_loss = tf.reduce_mean(normal_dist)
   # for i in range(len(side_list)):
   #     ctr_i = tf.gather(ctr, side_list[i] , axis=1)
   #     #pt_err_i = tf.gather(dist_pred2gt, cap_id_data[i], axis=1)
   #     #nrm_err_i = tf.gather(normal_dist, cap_id_data[i], axis=1)
   #     pt_err_i = tf.gather(dist_pred2gt, side_list[i], axis=1)
   #     nrm_err_i = tf.gather(normal_dist, side_list[i], axis=1)
   #     if if_mask:
   #         extra_pt_err += masked_loss(pt_err_i, ctr_i, gt_pt, tf.reduce_mean(pt_err_i))
   #         extra_nrm_err += masked_loss(nrm_err_i, ctr_i, gt_pt,tf.reduce_mean(nrm_err_i))
   #     else:
   #         extra_pt_err += tf.reduce_mean(pt_err_i) 
   #         extra_nrm_err += tf.reduce_mean(nrm_err_i)
        
    point_loss = (2.-cf_ratio)*tf.reduce_mean(dist_gt2pred) + cf_ratio* (pred2gt_err + extra_pt_err * 2.)
    normal_loss = (2.-cf_ratio)*tf.reduce_mean(normal_dist_2) + cf_ratio * (normal_loss + extra_nrm_err * 2.)

    return point_loss, normal_loss, edge_loss, lap_loss, cap_loss

def masked_loss(pred, pred_ctr, gt_pt, bar):
    mask = tf.where(tf.logical_and(tf.reduce_all(pred_ctr<tf.reduce_max(gt_pt, axis=1, keepdims=True),-1), tf.reduce_all(pred_ctr>tf.reduce_min(gt_pt, axis=1, keepdims=True), -1)))
    pred_bounded = tf.gather_nd(pred, mask, batch_dims=0)
    err = tf.where(tf.shape(pred_bounded)[-1]>0 , tf.reduce_mean(pred_bounded), bar)
    return err

def calc_cap_normal_loss(pred_nrm, ctr_list, side_list, cap_id_data, mesh_id):
    loss = 0.
    for i in range(len(ctr_list)):
        ctr_nrms = tf.gather(pred_nrm, ctr_list[i], axis=1)
        ctr_nrms = tf.reduce_mean(ctr_nrms, axis=1)
        nrms_dot = tf.reduce_sum(tf.square(ctr_nrms - pred_nrm), axis=-1)
        nrms_dot = tf.gather(nrms_dot, cap_id_data[i], axis=1)
        side_nrms = tf.gather(pred_nrm, side_list[i], axis=1)
        side_nrms_dot = tf.abs(tf.reduce_sum(ctr_nrms * side_nrms, axis=-1))
        loss += tf.reduce_mean(nrms_dot) + tf.reduce_mean(side_nrms_dot) * 0.5
    return loss

def mesh_loss(pred, gt, feed_dict, mesh_id, cf_ratio=1., if_mask=False, if_l2=True):
    #weights = [0.1, 0.6, 0.3]
    #weights = [1., 0.5, 0.5]
    weights = [0.5, 0.5, 0.5]
    gt_pt = gt[:, :, :3] # gt points
    gt_nm = gt[:, :, 3:] # gt normals
    
    mesh_n = feed_dict['struct_node_ids'][mesh_id+1] - feed_dict['struct_node_ids'][mesh_id]
    mesh = pred[:, :mesh_n, :]
    sample = pred[:, mesh_n:, :]
    ctr_list = feed_dict['ctr_data'][mesh_id]
    side_list = feed_dict['side_data'][mesh_id]
    cap_id_data = feed_dict['cap_data'][mesh_id]

    point_loss_p, normal_loss_p, edge_loss_p, lap_loss_p, cap_loss_p = mesh_loss_single(mesh, gt, feed_dict['tmplt_faces'][mesh_id], feed_dict['lap_list'][mesh_id], ctr_list, side_list, cap_id_data, cf_ratio, if_mask, mesh_id=mesh_id)
    point_loss_s, normal_loss_s, edge_loss_s, lap_loss_s, cap_loss_s = mesh_loss_single(sample, gt, feed_dict['sample_faces'][mesh_id], feed_dict['sample_lap_list'][mesh_id], ctr_list, side_list, cap_id_data, cf_ratio, if_mask, mesh_id=mesh_id)
    #corr_sample = tf.gather(mesh, feed_dict['id_mesh_on_sample'][mesh_id], axis=1)
    #sample_mesh_dist = tf.reduce_mean(tf.square(corr_sample - sample))
    sample_mesh_dist = tf.reduce_mean(tf.square(mesh - sample))
    point_total = point_loss_p * weights[0] + point_loss_s * weights[1]
    normal_total = normal_loss_p * weights[0] + normal_loss_s * weights[1]
    edge_total = edge_loss_p * weights[0] + edge_loss_s * weights[1]
    lap_total = lap_loss_p * weights[0] + lap_loss_s * weights[1]
    cap_total = cap_loss_p * weights[0] + cap_loss_s * weights[1]
    if if_l2:
        print("L2 loss!")
        point_total += sample_mesh_dist * weights[2]
    return point_total, normal_total, edge_total, lap_total, cap_total

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score
  
  
def dice_loss(y_true, y_pred):
    num_class = y_pred.get_shape().as_list()[-1]
    y_true_one_hot = tf.one_hot(tf.cast(y_true,tf.int32), num_class)
    loss = 0.
    for i in range(num_class):
        loss += (1 - dice_coeff_mean(y_true_one_hot[:,:,:,:,:,i], y_pred[:,:,:,:,i]))
    return loss


def dice_coeff_mean(y_true, y_pred):
    smooth = 1.
    # Flatten
    shape = tf.shape(y_pred)
    batch = shape[0]
    length = tf.reduce_prod(shape[1:])
    y_true_f = tf.cast(tf.reshape(y_true, [batch,length]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [batch,length]), tf.float32)
    intersection = tf.reduce_sum(tf.multiply(y_true_f ,y_pred_f), axis=-1)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=-1) + tf.reduce_sum(y_pred_f, axis=-1) + smooth)
    return tf.reduce_mean(score)

def bce_dice_loss(y_true, y_pred):
    loss = losses.sparse_categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
  
def binary_bce_dice_loss(y_true, y_pred):
    condition = tf.greater(y_true, 0)
    res = tf.where(condition, tf.ones_like(y_true), y_true)
    pred = tf.sigmoid(y_pred)
    pred = tf.clip_by_value(pred, 1e-6, 1.-1e-6)
    loss = losses.binary_crossentropy(res, pred) + (1-dice_coeff_mean(res, pred))
    return loss 
    

