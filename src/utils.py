import os
import numpy as np
import glob
import re
try:
    import tensorflow as tf
    from tensorflow.python.keras import backend as K
    from dataset import _parse_function_all
except Exception as e: print(e)

def positional_encoding(max_position, d_model, min_freq=1e-4):
    print("Positional encoding for {} nodes with dim of {}".format(max_position, d_model))
    position = np.arange(max_position)
    freqs = min_freq**(2*(np.arange(d_model)//2)/d_model)
    pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)
    pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
    return pos_enc

def fourier_feature_mapping(num_features, coords, scale=1., seed=42):
    print("Fourier feature mapping: ", coords.shape)
    np.random.seed(seed)
    B_mat = scale * np.random.normal(size=(coords.shape[-1], num_features))
    rff_input = np.concatenate([np.sin((2*np.pi*coords) @ B_mat), np.cos((2*np.pi*coords) @ B_mat)], axis=-1)
    return rff_input

def dice_score(pred, true):
    pred = pred.astype(np.int)
    true = true.astype(np.int)
    num_class = np.unique(true)

    #change to one hot
    dice_out = [None]*len(num_class)
    for i in range(1, len(num_class)):
        pred_c = pred == num_class[i]
        true_c = true == num_class[i]
        dice_out[i] = np.sum(pred_c*true_c)*2.0 / (np.sum(pred_c) + np.sum(true_c))

    mask =( pred > 0 )+ (true > 0)
    dice_out[0] = np.sum((pred==true)[mask]) * 2. / (np.sum(pred>0) + np.sum(true>0))
    return dice_out

import csv
def write_scores(csv_path,scores): 
    with open(csv_path, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(('Dice', 'ASSD'))
        for i in range(len(scores)):
            writer.writerow(tuple(scores[i]))
            print(scores[i])
    writeFile.close()
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def buildImageDataset(data_folder_out, modality, seed, mode='_train', ext='*.tfrecords'):
    import random
    x_train_filenames = []
    filenames = [None]*len(modality)
    nums = np.zeros(len(modality))
    for i, m in enumerate(modality):
      filenames[i], _ = getTrainNLabelNames(data_folder_out, m, ext=ext, fn=mode)
      nums[i] = len(filenames[i])
      x_train_filenames+=filenames[i]
      #shuffle
      random.shuffle(x_train_filenames)
    random.shuffle(x_train_filenames)      
    print("Number of images obtained for training and validation: " + str(nums))
    return x_train_filenames

def construct_feed_dict(pkl, num_block, has_cap=True):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict['image_data'] = None
    feed_dict['mesh_coords'] = tf.convert_to_tensor(pkl['tmplt_coords'], dtype=tf.float32)
    feed_dict['sample_coords'] = tf.convert_to_tensor(pkl['sample_coords'], dtype=tf.float32)
    feed_dict['pe'] = tf.convert_to_tensor(fourier_feature_mapping(384, pkl['sample_coords'], scale=1., seed=42), dtype=tf.float32)
    #feed_dict['pe'] = tf.convert_to_tensor(fourier_feature_mapping(192, pkl['sample_coords'], scale=1., seed=42), dtype=tf.float32)
    feed_dict['adjs'] = [tf.SparseTensor(indices=j[0], values=j[1].astype(np.float32), dense_shape=j[-1]) for j in pkl['support']]
    feed_dict['struct_node_ids'] = pkl['struct_node_ids']
    feed_dict['sample_node_ids'] = pkl['sample_node_list']
    feed_dict['sample_lap_list'] = [tf.convert_to_tensor(i, dtype=tf.int32) for i in pkl['sample_lap_list']]
    feed_dict['lap_list'] = [tf.convert_to_tensor(i, dtype=tf.int32) for i in pkl['lap_list']]

    ## Find the unique face ids and sort based on node id
    #for i, (faces, sample_faces) in enumerate(zip(pkl['tmplt_faces'], pkl['sample_faces'])):
    #    _, index = np.unique(faces[:,0], return_index=True)
    #    pkl['tmplt_faces'][i] = faces[index, :]
    #    _, index = np.unique(faces[:,0], return_index=True)
    #    pkl['sample_faces'][i] = sample_faces[index, :]
    #feed_dict['id_ctrl_on_sample'] = pkl['id_ctrl_on_sample']
    feed_dict['lap_ids'] = tf.convert_to_tensor(pkl['lap_ids'], dtype=tf.int32)
    feed_dict['tmplt_faces'] = [tf.convert_to_tensor(faces, dtype=tf.int32) for faces in pkl['tmplt_faces']]
    for i in range(len(feed_dict['tmplt_faces'])):
        #print(pkl['tmplt_faces'][i])
        #print("UTILS DEBUG: ", feed_dict['tmplt_faces'][i].get_shape().as_list(), pkl['tmplt_faces'][i].shape())
        print("UTILS DEBUG: ", feed_dict['tmplt_faces'][i].get_shape().as_list())
    feed_dict['sample_faces'] = [tf.convert_to_tensor(sample_faces, dtype=tf.int32) for sample_faces in pkl['sample_faces']]
    feed_dict['id_mesh_on_sample'] = [tf.convert_to_tensor(i, dtype=tf.int32) for i in pkl['id_mesh_on_sample']]
    feed_dict['id_ctrl_on_sample'] = [tf.convert_to_tensor(i, dtype=tf.int32) for i in pkl['id_ctrl_on_sample_all']]
    feed_dict['bbw'] = [tf.convert_to_tensor(i, dtype=tf.float32) for i in pkl['bbw']]
    if len(feed_dict['bbw']) == 1:
        feed_dict['id_ctrl_on_sample'] = feed_dict['id_ctrl_on_sample']*num_block
        feed_dict['bbw'] = feed_dict['bbw']*num_block
    if has_cap:
        feed_dict['cap_data'] = []
        for c_id in pkl['cap_data']:
            sub_list = []
            for q in np.unique(c_id):
                if q > 0:
                    sub_list.append(tf.convert_to_tensor(np.where(np.array(c_id)==q)[0], dtype=tf.int32))
            feed_dict['cap_data'].append(sub_list)
        print("feed_dict['cap_data']: ", feed_dict['cap_data'])
        #pkl['cap_ctr_data'] = [[[-1]], [[1802]], [[-1]], [[89]], [[-1]], [[5954]], [[-1]]]
        feed_dict['ctr_data'] = [[tf.convert_to_tensor(ctr_id, dtype=tf.int32) for ctr_id in ctr_id_list if len(ctr_id)>0] for ctr_id_list in pkl['cap_ctr_data']]
        feed_dict['side_data'] = [[tf.convert_to_tensor(side_id, dtype=tf.int32) for side_id in side_id_list if len(side_id)>0] for side_id_list in pkl['cap_side_data']]
    else:
        feed_dict['cap_data'] = [[]]*len(feed_dict['tmplt_faces'])
        feed_dict['ctr_data'] = [[]]*len(feed_dict['tmplt_faces']) 
        feed_dict['side_data'] = [[]]*len(feed_dict['tmplt_faces']) 

    return feed_dict

def construct_feed_dict_ffd(pkl, if_im=False, if_random=False):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict['image_data'] = None
    feed_dict['ffd_matrix_image'] = None
    #feed_dict['tmplt_coords'] = pkl['tmplt_coords'].astype(np.float32)
    feed_dict['grid_coords'] = tf.convert_to_tensor(pkl['grid_coords'], dtype=tf.float32)
    feed_dict['sample_coords'] = tf.convert_to_tensor(pkl['sample_coords'], dtype=tf.float32)
    feed_dict['ffd_matrix_mesh'] = [tf.SparseTensor(indices=i[0], values=i[1].astype(np.float32), dense_shape=i[-1]) for i in pkl['ffd_matrix_mesh']]
    feed_dict['grid_downsample'] = [tf.SparseTensor(indices=i[0], values=i[1].astype(np.float32), dense_shape=i[-1]) for i in pkl['grid_downsample']]
    feed_dict['grid_upsample'] = [tf.SparseTensor(indices=i[0], values=i[1].astype(np.float32), dense_shape=i[-1]) for i in pkl['grid_upsample']]
    feed_dict['grid_size'] = [int(round(i[-1][-1]**(1/3))) for i in pkl['ffd_matrix_mesh']]
    feed_dict['struct_node_ids'] = pkl['struct_node_ids']

    # Find the unique face ids and sort based on node id
    for i, faces in enumerate(pkl['tmplt_faces']):
        _, index = np.unique(faces[:,0], return_index=True)
        pkl['tmplt_faces'][i] = faces[index, :]

    feed_dict['tmplt_faces'] = [tf.convert_to_tensor(faces, dtype=tf.int32) for faces in pkl['tmplt_faces']]
    #feed_dict['adjs']= [tf.SparseTensor(indices=j[0], values=j[1].astype(np.float32), dense_shape=j[-1]) for j in pkl['support']]
    feed_dict['adjs']= [[tf.SparseTensor(indices=j[0], values=j[1].astype(np.float32), dense_shape=j[-1]) for j in l] for l in pkl['support']]
    return feed_dict

def getTrainNLabelNames(data_folder, m, ext='*.nii.gz',fn='_train', seg_fn='_masks'):
  x_train_filenames = []
  y_train_filenames = []
  for subject_dir in sorted(glob.glob(os.path.join(data_folder,m+fn,ext))):
      x_train_filenames.append(os.path.realpath(subject_dir))
  try:
      for subject_dir in sorted(glob.glob(os.path.join(data_folder ,m+fn+seg_fn,ext))):
          y_train_filenames.append(os.path.realpath(subject_dir))
  except Exception as e: print(e)

  return x_train_filenames, y_train_filenames

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def data_to_tfrecords(X, Y, S,transform, spacing, file_path_prefix=None, verbose=True, debug=True):
           
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
           
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print("Serializing example into {}".format(result_tf_file))
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    
    d_feature = {}
    d_feature['X'] = _float_feature(X.flatten())
    d_feature['S'] = _int64_feature(S.flatten())

    if debug:
        print("**** X ****")
        print(X.shape, X.flatten().shape)
        print(X.dtype)
    for i, y in enumerate(Y):
        d_feature['Y_'+str(i)] = _float_feature(y.flatten())
        if debug:
            print("**** Y shape ****")
            print(y.shape, y.flatten().shape)
            print(y.dtype)

    d_feature['Transform'] = _float_feature(transform.flatten())
    d_feature['Spacing'] = _float_feature(spacing)
    #first axis is the channel dimension
    d_feature['shape0'] = _int64_feature([X.shape[0]])
    d_feature['shape1'] = _int64_feature([X.shape[1]])    
    d_feature['shape2'] = _int64_feature([X.shape[2]])

    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)
    
    if verbose:
        print("Writing {} done!".format(result_tf_file))

