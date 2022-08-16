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
    tf.ceil = tf.math.ceil
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tf_utils
import numpy as np

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
#        res = tf.sparse_tensor_dense_matmul(x, y)
        res = tf_utils.sparse_tensor_dense_tensordot(x, y, axes=[[1], [1]])
        res = tf.transpose(res, perm=[1,0,2])
    else:
#        res = tf.matmul(x, y)
        res = tf.tensordot(x, y, axes=1)
        #res = tf.transpose(res, perm=[1,0,2])
    return res

from math import factorial

class BoundaryMask(layers.Layer):

    def __init__(self, **kwargs):
        super(BoundaryMask, self).__init__(**kwargs)
    def get_config(self):
        base_config = super(BoundaryMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def call(self, inputs):
        target, source, source_coords = inputs
        mask = tf.logical_and(tf.reduce_all(source_coords>0.05, axis=-1, keepdims=True), tf.reduce_all(source_coords<0.95, axis=-1, keepdims=True))
        dim = source.get_shape().as_list()[-1]
        source = tf.where(tf.tile(mask, [1, 1, dim]), source, target)
        return source


class UNet3DIsensesConvBlock(layers.Layer):

    def __init__(self, num_filters, strides=(1,1,1), **kwargs):
        super(UNet3DIsensesConvBlock, self).__init__(**kwargs)
        print("ConvBlock: ", num_filters, strides)
        self.conv = layers.Conv3D(num_filters, (3, 3, 3), kernel_regularizer=regularizers.l2(0.01),  padding='same', strides=strides)
        self.inst = InstanceNormalization(axis=-1)
        self.act = layers.LeakyReLU(alpha=0.01)
        self.num_filters = num_filters
        self.strides = strides
    
    def get_config(self):
        config = {'num_filters': self.num_filters, 'strides': self.strides}
        base_config = super(UNet3DIsensesConvBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        output = self.conv(inputs)
        output = self.act(self.inst(output))
        return output

class UNet3DIsensesContextModule(layers.Layer):

    def __init__(self, num_filters, strides=(1,1,1), dropout_rate=0.3, **kwargs):
        super(UNet3DIsensesContextModule, self).__init__(**kwargs)
        print(strides, num_filters)
        self.conv_bl_0 = UNet3DIsensesConvBlock(num_filters, strides=strides)
        self.conv_bl_1 = UNet3DIsensesConvBlock(num_filters)
        self.dropout = layers.SpatialDropout3D(rate=dropout_rate)
        self.conv_bl_2 = UNet3DIsensesConvBlock(num_filters)
        self.add = layers.Add()
        self.num_filters = num_filters
        self.strides = strides
        self.dropout_rate = dropout_rate
    
    def get_config(self):
        config = {'num_filters': self.num_filters, 'strides': self.strides, 'dropout_rate': self.dropout_rate}
        base_config = super(UNet3DIsensesContextModule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        print("UNet3DIsensesContextModule: ", inputs)
        conv_0 = self.conv_bl_0(inputs)
        print("UNet3DIsensesContextModule: ", conv_0)
        conv_1 = self.conv_bl_1(conv_0)
        dropout = self.dropout(conv_1)
        conv_2 = self.conv_bl_2(dropout)
        sum_output = self.add([conv_0, conv_2])
        print("UNet3DIsensesContextModule: ", sum_output)
        return sum_output

class UNet3DIsenseeEncoder(layers.Layer):

    def __init__(self, num_filters=[16, 32, 64, 128, 256], strides=[(1,1,1), (2,2,2), (2,2,2), (2,2,2), (2,2,2)], **kwargs):
        super(UNet3DIsenseeEncoder, self).__init__(**kwargs)
        self.num_filters = num_filters
        self._context_module_list = []
        for f_num, s in zip(num_filters, strides):
            self._context_module_list.append(UNet3DIsensesContextModule(num_filters=f_num, strides=s)) 
        self.num_filters = num_filters
        self.strides = strides
    
    def get_config(self):
        config = {'num_filters': self.num_filters, 'strides': self.strides}
        base_config = super(UNet3DIsenseeEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, inputs):
        output_list = []
        outputs = inputs
        for m in self._context_module_list:
            outputs = m(inputs)
            output_list.append(outputs)
            inputs = outputs
        print("UNet3DIsenseeEncoder: ", output_list)
        return output_list

def comb(n, k):
    return factorial(n) / factorial(k) / factorial(n - k)

class Print(layers.Layer):
    def __init__(self, message, func, **kwargs):
        self.msg = message
        self.func = func
        super(Print, self).__init__(**kwargs)
    def call(self, x):
        values = [f(x) for f in self.func]
        values = tuple(values)
        tf.print(self.msg, *values)
        return x

class WeightedConcatenate(layers.Concatenate):
    def __init__(self, **kwargs):
        super(WeightedConcatenate, self).__init__(**kwargs)
    def build(self, input_shape):
        super(WeightedConcatenate, self).build(input_shape)
    def call(self, x):
        in1, in2 = x
        in1_mean = tf.reduce_mean(in1**2, axis=[1,2], keepdims=True)
        in2_mean = tf.reduce_mean(in2**2, axis=[1,2], keepdims=True)
        in2 = in2*tf.sqrt(in1_mean/in2_mean)
        out = tf.concat([in1, in2], axis=self.axis)
        return out

class MatMul(layers.Layer):
    def __init__(self,matrix, sparse=False, **kwargs):
        super(MatMul, self).__init__(**kwargs)
        self.matrix = matrix
        self.sparse = sparse
    def get_config(self):
        config = {'matrix': self.matrix, 'sparse': self.sparse}
        base_config = super(MatMul, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def build(self, input_shape):
        super(MatMul, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        if self.sparse:
            output = dot(self.matrix, x, self.sparse)
        else:
            matrix = tf.expand_dims(self.matrix, axis=0)
            output = tf.matmul(matrix, x)
        return output
    def compute_output_shape(self, input_shape):
        output_shape = tf.identity(input_shape)
        output_shape[1] = self.matrix.get_shape().as_list()[1]
        return output_shape 

class ExpandDim(layers.Layer):
    def __init__(self, axis=0,**kwargs):
        super(ExpandDim, self).__init__(**kwargs)
        self.axis = axis
    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(ExpandDim, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def call(self, x):
        x = tf.expand_dims(x, axis=self.axis)
        return x
    def compute_output_shape(self, input_shape):
        if self.axis==-1:
            input_shape = list(input_shape).append(1)
        else:
            input_shape = list(input_shape).insert(self.axis, 1)
        return input_shape

class FFD(layers.Layer):
    def __init__(self,ffd_matrix,scale_vec=None, offset=None, **kwargs):
        super(FFD, self).__init__(**kwargs)
        self.ffd_matrix = ffd_matrix
        self.scale_vec = tf.expand_dims(tf.constant(scale_vec, tf.float32), axis=0) if scale_vec is not None else None
        self.offset = tf.expand_dims(tf.constant(offset, tf.float32), axis=0) if offset is not None else None
    def get_config(self):
        config = {'ffd_matrix': self.ffd_matrix, 'scale_vec': self.scale_vec,
                'offset': self.offset}
        base_config = super(FFD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def build(self, input_shape):
        self.batch_size = input_shape[0]
        super(FFD, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        if self.scale_vec is not None:
            x = x * self.scale_vec
        if self.offset is not None:
            x = x - self.offset
        dx = dot(self.ffd_matrix, x, sparse=True)
        return dx
    def compute_output_shape(self, input_shape):
        output_shape = tf.identity(input_shape)
        output_shape[1] = self.ffd_matrix.get_shape().as_list()[0]
        return output_shape 

class Tile(layers.Layer):
    def __init__(self, repeats,**kwargs):
        super(Tile, self).__init__(**kwargs)
        self.repeats = repeats
    def get_config(self):
        config = {'repeats': self.repeats}
        base_config = super(Tile, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def call(self, x):
        x = tf.tile(x, self.repeats)
        return x

class Split(layers.Layer):
    def __init__(self, axis=-1, num=1,**kwargs):
        super(Split, self).__init__(**kwargs)
        self.axis = axis
        self.num = num
    def get_config(self):
        config = {'axis': self.axis, 'num': self.num}
        base_config = super(Split, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def build(self, input_shape):
        super(Split, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        x = tf.split(x, self.num, axis=self.axis)
        return x
    def compute_output_shape(self, input_shape):
        shape_list = []
        input_shape[self.axis] = input_shape[self.axis] // self.num
        for i in range(self.num):
            shape_list.append(input_shape)
        return shape_list

class ScalarMul(layers.Layer):
    def __init__(self, factor=1.,**kwargs):
        super(ScalarMul, self).__init__(**kwargs)
        self.factor = factor
    def get_config(self):
        config = {'factor': self.factor}
        base_config = super(ScalarMul, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def call(self, x):
        x *= self.factor
        return x

def gather_nd(features, indices):
    # tf1.12 does not support gather_nd with batch_dims; work around: 
    ind_shape = tf.shape(indices)
    indices = tf.reshape(indices, [ind_shape[0]*ind_shape[1], ind_shape[2]])
    first = tf.cast(tf.range(tf.size(indices[:,0]))/ind_shape[1], dtype=tf.int32)
    indices = tf.concat([tf.expand_dims(first, axis=-1), indices], axis=1)
    gather = tf.reshape(tf.gather_nd(features, indices), [ind_shape[0],ind_shape[1],tf.shape(features)[-1]])
    return gather

class Projection(layers.Layer):
    def __init__(self, feature_block_ids=[1], size=128, **kwargs):
        super(Projection, self).__init__(**kwargs)
        self.feature_block_ids = feature_block_ids
        self.size = size
    def get_config(self):
        config = {'feature_block_ids': self.feature_block_ids, 'size': self.size}
        base_config = super(Projection, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # input shapes: 4 by 4 transform matrix, feature 1, 2, 3, 4, mesh_coords
        assert isinstance(input_shape, list)
        self.batch_size = input_shape[-1][0]
        super(Projection, self).build(input_shape)  # Be sure to call this at the end
    
    def call(self, inputs):
        mesh_coords = inputs[-1]
        features = inputs[:-1]
        mesh_shape = mesh_coords.get_shape().as_list()
        mesh_coords = tf.reshape(mesh_coords, [mesh_shape[0], mesh_shape[1]*(mesh_shape[2]//3), 3])
        out = tf.zeros([mesh_shape[0], mesh_shape[1]*(mesh_shape[2]//3), 0], tf.float32)
        num = len(features)
        id_list = self.feature_block_ids
        features = [features[i] for i in self.feature_block_ids]
        for i, power in enumerate(id_list):
            factor = tf.constant([[[(0.5**power)*self.size[0], (0.5**power)*self.size[1], (0.5**power)*self.size[2]]]], dtype=tf.float32)
            factor = tf.tile(factor, [tf.shape(mesh_coords)[0], 1,1])
            indices = mesh_coords * factor
            #indices = tf.cast(indices[:,:,:-1], tf.int32)
            #indices = tf.Print(indices, [self.feature_block_ids, indices, mesh_coords], message='Level: %d' % i)
            indices = tf.clip_by_value(indices, 0.01,tf.cast(tf.reduce_min(tf.shape(features[i])[1:4]), tf.float32)-1.01)
            x1 = tf.floor(indices[:,:,0])
            x2 = tf.ceil(indices[:,:,0])
            y1 = tf.floor(indices[:,:,1])
            y2 = tf.ceil(indices[:,:,1])
            z1 = tf.floor(indices[:,:,2])
            z2 = tf.ceil(indices[:,:,2])
            q11 = gather_nd(features[i], tf.cast(tf.stack([x1, y1, z1], axis=-1), tf.int32))
            #q11 = tf.Print(q11, [indices, q11])
            q21 = gather_nd(features[i], tf.cast(tf.stack([x2, y1, z1], axis=-1), tf.int32))
            q12 = gather_nd(features[i], tf.cast(tf.stack([x1, y2, z1], axis=-1), tf.int32))
            q22 = gather_nd(features[i], tf.cast(tf.stack([x2, y2, z1], axis=-1), tf.int32))
            wx = tf.expand_dims(tf.subtract(indices[:,:,0], x1), -1)
            wx2 = tf.expand_dims(tf.subtract(x2, indices[:,:,0]), -1)
            lerp_x1 = tf.add(tf.multiply(q21, wx), tf.multiply(q11, wx2))
            lerp_x2 = tf.add(tf.multiply(q22, wx), tf.multiply(q12, wx2))
            wy = tf.expand_dims(tf.subtract(indices[:,:,1], y1), -1)
            wy2 = tf.expand_dims(tf.subtract(y2, indices[:,:,1]), -1)
            lerp_y1 = tf.add(tf.multiply(lerp_x2, wy), tf.multiply(lerp_x1, wy2))

            q11 = gather_nd(features[i], tf.cast(tf.stack([x1, y1, z2], axis=-1), tf.int32))
            q21 = gather_nd(features[i], tf.cast(tf.stack([x2, y1, z2], axis=-1), tf.int32))
            q12 = gather_nd(features[i], tf.cast(tf.stack([x1, y2, z2], axis=-1), tf.int32))
            q22 = gather_nd(features[i], tf.cast(tf.stack([x2, y2, z2], axis=-1), tf.int32))
            lerp_x1 = tf.add(tf.multiply(q21, wx), tf.multiply(q11, wx2))
            lerp_x2 = tf.add(tf.multiply(q22, wx), tf.multiply(q12, wx2))
            lerp_y2 = tf.add(tf.multiply(lerp_x2, wy), tf.multiply(lerp_x1, wy2))

            wz = tf.expand_dims(tf.subtract(indices[:,:,2], z1), -1)
            wz2 = tf.expand_dims(tf.subtract(z2, indices[:,:,2]),-1)
            lerp_z = tf.add(tf.multiply(lerp_y2, wz), tf.multiply(lerp_y1, wz2))
            out = tf.concat([out, lerp_z], axis=-1)
        #out -= tf.reduce_mean(out, [1,2], keepdims=True)
        out = tf.reshape(out, [mesh_shape[0], mesh_shape[1], out.get_shape().as_list()[-1]*(mesh_shape[2]//3)])
        return out

class GraphConv(layers.Layer):
    def __init__(self, input_dim=10, output_dim=10, adjs=None, dropout=False,
                 sparse_inputs=False, act=tf.nn.relu, bias=True,
                 featureless=False, **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.act = act
        self.featureless = featureless
        self.vars = {}
        self.adjs = adjs
    def get_config(self):
        config = {'input_dim': self.input_dim, 
                'output_dim': self.output_dim, 
                'adjs': self.adjs,
                'dropout':self.dropout, 
                'sparse_inputs': self.sparse_inputs,
                'act': self.act, 
                'bias':self.bias, 
                'featureless': self.featureless}
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.batch_size = input_shape[0]
        num_cheb_support = len(self.adjs)
        for i in range(1, num_cheb_support+1):
            name = 'kernel_'+str(i)
            self.vars[name] = self.add_weight(name=name, 
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_normal',
                                      regularizer=regularizers.l2(0.01), 
                                      trainable=True)
        self.vars['bias'] = self.add_weight(name='bias', 
                                      shape=( self.output_dim),
                                      initializer='zeros',
                                      #regularizer=regularizers.l2(0.01), 
                                      trainable=True)
        super(GraphConv, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        support_1 = dot(x, self.vars['kernel_1'], sparse=self.sparse_inputs)
        output = dot(self.adjs[0], support_1, sparse=True)
        for i in range(2, len(self.adjs)+1):
            name = 'kernel_'+str(i)
            support = dot(x, self.vars[name], sparse=self.sparse_inputs)
            output = output + dot(self.adjs[i-1], support, sparse=True)
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

    def compute_output_shape(self, input_shape):
        output_shape = tf.identity(input_shape)
        output_shape[-1] = self.output_dim
        return output_shape

from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.python.keras import initializers, regularizers, constraints

class InstanceNormalization(layers.Layer):
    """Instance normalization layer. Taken from keras.contrib
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

