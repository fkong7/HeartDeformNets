import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from custom_layers import *
import numpy as np
import sys
from utils import positional_encoding

class HeartDeformNet(object):
    def __init__(self, batch_size, input_size, hidden_dim, feed_dict,amplify_factor=1., num_mesh=1, num_seg=1, num_block=3, train=False):
        super(HeartDeformNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.batch_size = batch_size
        self.feed_dict = feed_dict
        self.amplify_factor = amplify_factor
        self.num_mesh = num_mesh
        self.num_seg = num_seg
        self.num_block = num_block
        self.train = train
        
    def build_3dunet(self):
        image_inputs = layers.Input(self.input_size, batch_size=self.batch_size)
        features =  self._unet_isensee_encoder(image_inputs, num_class=self.num_seg)
        output = self._unet_isensee_decoder(features, num_filters=[128, 64, 32, 16], num_class=self.num_seg)
        return models.Model([image_inputs], outputs)

    def build_bc(self):
        image_inputs = layers.Input(self.input_size, batch_size=self.batch_size)
        features =  self._unet_isensee_encoder(image_inputs, num_class=self.num_seg) 
        if self.num_seg >0:
            decoder =  self._unet_isensee_decoder(features)
        
        mesh_coords = layers.Lambda(lambda x: self.feed_dict['mesh_coords'])(image_inputs)
        sample_coords = layers.Lambda(lambda x: self.feed_dict['sample_coords'])(image_inputs)
        pe = layers.Lambda(lambda x: self.feed_dict['pe'])(image_inputs)
        print("POSITIONAL ENCODING: ", pe.get_shape().as_list())
        #print("Grid coords: ", grid_coords.get_shape().as_list())
        mesh_coords_p = ExpandDim(axis=0)(mesh_coords) 
        mesh_coords_p = Tile((self.batch_size, 1, 1))(mesh_coords_p)
        sample_coords_p = ExpandDim(axis=0)(sample_coords) 
        sample_coords_p = Tile((self.batch_size, 1, 1))(sample_coords_p)
        pe_p = ExpandDim(axis=0)(pe) 
        pe_p = Tile((self.batch_size, 1, 1))(pe_p)
        outputs = self._bc_decoder((features, mesh_coords_p, sample_coords_p, pe_p), self.hidden_dim)
        if self.num_seg >0:
            outputs = [decoder]+ list(outputs)
        return models.Model([image_inputs],outputs)

    def build_ffd(self):
        image_inputs = layers.Input(self.input_size, batch_size=self.batch_size)
        features =  self._unet_isensee_encoder(image_inputs, num_class=self.num_seg) 
        if self.num_seg >0:
            decoder =  self._unet_isensee_decoder(features)

        grid_coords = layers.Lambda(lambda x: self.feed_dict['grid_coords'])(image_inputs)
        sample_coords = layers.Lambda(lambda x: self.feed_dict['sample_coords'])(image_inputs)
        adjs = self.feed_dict['adjs']
        ffd_matrix_mesh = self.feed_dict['ffd_matrix_mesh']
        ffd_matrix_image = self.feed_dict['ffd_matrix_image']
        grid_coords_p = ExpandDim(axis=0)(grid_coords) 
        grid_coords_p = Tile((self.batch_size, 1, 1))(grid_coords_p)
        sample_coords_p = ExpandDim(axis=0)(sample_coords) 
        sample_coords_p = Tile((self.batch_size, 1, 1))(sample_coords_p)
        outputs = self._ffd_decoder((image_inputs, features, grid_coords_p, sample_coords_p, ffd_matrix_mesh, ffd_matrix_image), self.hidden_dim, adjs)
        if self.num_seg >0:
            outputs = [decoder]+ list(outputs)
        return models.Model([image_inputs],outputs)

    def load_pre_trained_weights(self, new_model, old_model_fn, trainable=False):
        pre_trained_im = UNet3DIsensee(self.input_size, num_class=self.num_seg)
        pre_trained_im = pre_trained_im.build()
        c = 0
        for i, layer in enumerate(pre_trained_im.layers[:55]):
            if 'lambda' in new_model.layers[i].name:
                c += 1
            print(i, layer.name, new_model.layers[i].name, new_model.layers[i+c].name)
            weights = layer.get_weights()
            new_model.layers[i+c].set_weights(weights)
            new_model.layers[i+c].trainable = trainable
        del pre_trained_im
        return new_model

    def _unet_isensee_encoder(self, inputs, num_filters=[16, 32, 64, 128, 256], num_class=1):
        unet = UNet3DIsensee(self.input_size, num_class=num_class)
        output0 = unet._context_module(num_filters[0], inputs, strides=(1,1,1))
        output1 = unet._context_module(num_filters[1], output0, strides=(2,2,2))
        output2 = unet._context_module(num_filters[2], output1, strides=(2,2,2))
        output3 = unet._context_module(num_filters[3], output2, strides=(2,2,2))
        output4 = unet._context_module(num_filters[4], output3, strides=(2,2,2))
        return (output0, output1, output2, output3, output4)
    def _unet_isensee_decoder(self, inputs, num_filters=[64, 32, 16, 4], num_class=1):
        unet = UNet3DIsensee(self.input_size, num_class=num_class)
        output0, output1, output2, output3, output4 = inputs
        decoder0 = unet._decoder_block(num_filters[0], [output3, output4])
        decoder1 = unet._decoder_block(num_filters[1], [output2, decoder0])
        decoder2 = unet._decoder_block(num_filters[2], [output1, decoder1])
        decoder3 = unet._decoder_block_last_simple(num_filters[3], [output0, decoder2])
        output0 = layers.Conv3D(unet.num_class, (1, 1, 1))(decoder3)
        output1 = layers.Conv3D(unet.num_class, (1, 1, 1))(decoder2)
        output2_up = layers.UpSampling3D(size=(2,2,2))(layers.Conv3D(unet.num_class, (1, 1, 1))(decoder1))

        output_sum = layers.Add()([output2_up, output1])
        output_sum = layers.UpSampling3D(size=(2,2,2))(output_sum)
        output = layers.Add()([output_sum, output0])
        #output_sum = layers.Add()([output_sum, output0])
        #output = layers.Softmax()(output_sum)
        return output
    def _graph_res_block(self, inputs, adjs, in_dim, hidden_dim):
        output = GraphConv(in_dim ,hidden_dim, act=tf.nn.relu, adjs=adjs)(inputs)
        output2 = GraphConv(in_dim, hidden_dim, act=tf.nn.relu, adjs=adjs)(output)
        return layers.Average()([inputs, output2])

    def _graph_conv_block(self, inputs, adjs, feature_dim, hidden_dim, coord_dim, num_blocks):
        output = GraphConv(feature_dim, hidden_dim, act=tf.nn.relu, adjs=adjs)(inputs)
        output_cat = self._graph_res_block(output, adjs, hidden_dim, hidden_dim)
        for _ in range(num_blocks):
            output_cat = self._graph_res_block(output_cat, adjs, hidden_dim, hidden_dim)
        output = GraphConv(hidden_dim, coord_dim, act=lambda x: x, adjs=adjs)(output_cat)
        #output = GraphConv(hidden_dim, coord_dim, act=tf.nn.tanh)([output_cat]+[i for i in adjs])
        return output, output_cat
   
    def _bc_decoder(self, inputs,  hidden_dim):
        #import tensorflow_addons as tfa
        print("Predict displacements.")
        coord_dim = 3
        if self.num_block==3:
            feat_level = [[3,4], [2, 3], [0,1,2]] #
            #feat_level = [[3,4], [3,4], [3,4]]
            #mesh_feat_num = [256, 256, 256]
            #mesh_feat_merge_num = [256, 256, 256]
            mesh_feat_num = [384, 96, 48]
            mesh_feat_merge_num = [256, 64, 32]
        elif self.num_block==2:
            feat_level = [[3, 4], [0, 1, 2]]
            mesh_feat_num = [384, 112]
            mesh_feat_merge_num = [256, 128]
        elif self.num_block==1:
            feat_level = [[4,3,2,1,0]]
            mesh_feat_num = [32]
            mesh_feat_merge_num = [128]
        else:
            raise NotImplementedError
        features, mesh_coords, sample_coords, curr_feat = inputs
        input_size = [float(i) for  i in list(self.input_size)]
        curr_mesh = mesh_coords
        curr_sample_space = sample_coords

        out_list, out_d_grid_list, out_image, out_grid_list = [], [], [], []
        self.feed_dict['adjs']
        for l, (mesh_dim, mesh_m_dim, feat) in enumerate(zip(mesh_feat_num, mesh_feat_merge_num, feat_level)):
            if l ==0:
                output = curr_feat
            else:
                output =  GraphConv(curr_feat.get_shape().as_list()[-1], mesh_dim, act=tf.nn.relu, adjs=self.feed_dict['adjs'])(curr_feat)
                l = l - 1
            
            output_feat = Projection(feat, input_size)([i for i in features]+[curr_sample_space])
            output = WeightedConcatenate(axis=-1)([output_feat, output])
            
            ctrl_pt_dx, curr_feat = self._graph_conv_block(output, self.feed_dict['adjs'], output.get_shape().as_list()[-1], mesh_m_dim, 3, 3)
            
            ctrl_pt_dx_scaled = ScalarMul(self.amplify_factor)(ctrl_pt_dx) # scaled dx for both sampling pts and ctrl pts
            out_d_grid_list.append(ScalarMul(1./self.amplify_factor)(ctrl_pt_dx_scaled)) # for consistency with previous experiements
            curr_sample_space = layers.Add()([curr_sample_space, ctrl_pt_dx_scaled])
            ctrl_pt_only = layers.Lambda(lambda x: tf.gather(x, self.feed_dict['id_ctrl_on_sample'][l], axis=1))(curr_sample_space) # dx for control points only
            curr_mesh = MatMul(self.feed_dict['bbw'][l], sparse=False)(ctrl_pt_only)
            if self.train:
                curr_mesh_concat = layers.Concatenate(axis=1)([curr_mesh, curr_sample_space])
            else:
                curr_mesh_concat = layers.Lambda(lambda x: x)(curr_mesh)
            curr_mesh_scaled = ScalarMul(128)(curr_mesh_concat)
            output1_list = []
            for i in range(len(self.feed_dict['struct_node_ids'])-1):
                curr_mesh_i = layers.Lambda(lambda x: x[:, self.feed_dict['struct_node_ids'][i]:self.feed_dict['struct_node_ids'][i+1], :])(curr_mesh_scaled)
                if self.train:
                    mesh_n = self.feed_dict['mesh_coords'].get_shape().as_list()[0]
                    curr_samp_i = layers.Lambda(lambda x: x[:, mesh_n+self.feed_dict['sample_node_ids'][i]:mesh_n+self.feed_dict['sample_node_ids'][i+1], :])(curr_mesh_scaled)
                    curr_mesh_i = layers.Concatenate(axis=1)([curr_mesh_i, curr_samp_i])
                else:
                    curr_mesh_i = layers.Lambda(lambda x: x)(curr_mesh_i)
                output1_list.append(curr_mesh_i)
            out_list += output1_list
        out_list =  out_image + out_d_grid_list + out_list
        return out_list
    def _ffd_decoder(self, inputs,  hidden_dim, adjs):
        coord_dim = 3
        if self.num_block==3:
            graph_conv_num = [384, 96, 48]
            graph_block_num = [256, 64, 32]
            feat_level = [[3,4], [2, 3], [0,1,2]]
            feat_scale = [1., 1., 1.]
            sample_num = [256, 32, 16]
        elif self.num_block==2:
            graph_conv_num = [384, 112]
            graph_block_num = [256, 64]
            feat_level = [[3, 4], [0, 1, 2]]
            feat_scale = [1., 1.]
        elif self.num_block==1:
            graph_conv_num = [32]
            graph_block_num = [128]
            feat_level = [[4,3,2,1,0]]
            feat_scale = [1.]
        else:
            raise NotImplementedError
        image_inputs, features, grid_coords, sample_coords, ffd_matrix_mesh,  ffd_matrix_image = inputs
        input_size = [float(i) for  i in list(self.input_size)]
        curr_grid = grid_coords
        curr_feat = grid_coords
        curr_sample_space = sample_coords

        out_list, out_d_grid_list, out_image, out_grid_list = [], [], [], []
        for l, (conv_num, block_num, feat, feat_s) in enumerate(zip(graph_conv_num, graph_block_num, feat_level, feat_scale)):
            output =  GraphConv(curr_feat.get_shape().as_list()[-1], conv_num, act=tf.nn.relu, adjs=adjs[l])(curr_feat)
            output_feat = Projection(feat, input_size)([i for i in features]+[curr_sample_space])
            output_feat = MatMul(self.feed_dict['grid_downsample'][l]*feat_s, sparse=True)(output_feat)
            output = WeightedConcatenate(axis=-1)([output_feat, output])
            output1_dx, curr_feat = self._graph_conv_block(output, adjs[l], output.get_shape().as_list()[-1], block_num, coord_dim, 3)
            
            output1_dx_scaled = ScalarMul(self.amplify_factor)(output1_dx)
            out_d_grid_list.append(output1_dx)
            if 0 < l : # if middle blocks, add additional deformation to mesh using down-sampled grid
                mesh1 = layers.Add()([mesh1, FFD(ffd_matrix_mesh[l])(output1_dx_scaled)])
            else:
                curr_grid = layers.Add()([curr_grid, output1_dx_scaled])
                mesh1 = FFD(ffd_matrix_mesh[l])(curr_grid)
            curr_sample_space = mesh1
            # upsample and prepare for next block
            if l < self.num_block - 1:
                curr_feat = MatMul(self.feed_dict['grid_upsample'][l], sparse=True)(curr_feat) 
            mesh1_scaled = ScalarMul(128)(mesh1)
            if self.num_mesh > 1:
                output1_list = []
                for i in range(len(self.feed_dict['struct_node_ids'])-1):
                    mesh1_i = layers.Lambda(lambda x: x[:, self.feed_dict['struct_node_ids'][i]:self.feed_dict['struct_node_ids'][i+1], :])(mesh1_scaled)
                    output1_list.append(mesh1_i)
                out_list += output1_list
            else:
                mesh_i = layers.Lambda(lambda x: x)(mesh1_scaled)
                out_list.append(mesh_i)
        out_list =  out_image + out_d_grid_list + out_list
        return out_list    

class UNet3DIsensee(object):
    def __init__(self, input_size, num_class=8, num_filters=[16, 32, 64, 128, 256]):
        super(UNet3DIsensee, self).__init__()
        self.num_class = num_class
        self.input_size = input_size
        self.num_filters = num_filters
    
    def build(self):
        inputs = layers.Input(self.input_size)

        output0 = self._context_module(self.num_filters[0], inputs, strides=(1,1,1))
        output1 = self._context_module(self.num_filters[1], output0, strides=(2,2,2))
        output2 = self._context_module(self.num_filters[2], output1, strides=(2,2,2))
        output3 = self._context_module(self.num_filters[3], output2, strides=(2,2,2))
        output4 = self._context_module(self.num_filters[4], output3, strides=(2,2,2))
        
        decoder0 = self._decoder_block(self.num_filters[3], [output3, output4])
        decoder1 = self._decoder_block(self.num_filters[2], [output2, decoder0])
        decoder2 = self._decoder_block(self.num_filters[1], [output1, decoder1])
        decoder3 = self._decoder_block_last(self.num_filters[0], [output0, decoder2])
        output0 = layers.Conv3D(self.num_class, (1, 1, 1))(decoder3)
        output1 = layers.Conv3D(self.num_class, (1, 1, 1))(decoder2)
        output2_up = layers.UpSampling3D(size=(2,2,2))(layers.Conv3D(self.num_class, (1, 1, 1))(decoder1))

        output_sum = layers.Add()([output2_up, output1])
        output_sum = layers.UpSampling3D(size=(2,2,2))(output_sum)
        output_sum = layers.Add()([output_sum, output0])
        output = layers.Softmax()(output_sum)

        return models.Model(inputs=[inputs], outputs=[output])

    def _conv_block(self, num_filters, inputs, strides=(1,1,1)):
        output = layers.Conv3D(num_filters, (3, 3, 3),kernel_regularizer=regularizers.l2(0.01),  padding='same', strides=strides)(inputs)
        output = layers.LeakyReLU(alpha=0.01)(InstanceNormalization(axis=-1)(output))
        return output

    def _context_module(self, num_filters, inputs, dropout_rate=0.3, strides=(1,1,1)):
        conv_0 = self._conv_block(num_filters, inputs, strides=strides)
        conv_1 = self._conv_block(num_filters, conv_0)
        dropout = layers.SpatialDropout3D(rate=dropout_rate)(conv_1)
        conv_2 = self._conv_block(num_filters, dropout)
        sum_output = layers.Add()([conv_0, conv_2])
        return sum_output
    
    def _decoder_block(self, num_filters,  inputs, strides=(2,2,2)):
        features, encoder_out = inputs
        upsample = layers.UpSampling3D(size=strides)(encoder_out)
        conv_1 = self._conv_block(num_filters, upsample)
        concat = layers.Concatenate(axis=-1)([conv_1, features])
        conv_2 = self._conv_block(num_filters, concat)
        conv_3 = layers.Conv3D(num_filters, (1,1,1), padding='same')(conv_2)
        output = layers.LeakyReLU(alpha=0.01)(InstanceNormalization(axis=-1)(conv_3))
        return output
    
    def _decoder_block_last_simple(self, num_filters,  inputs, strides=(2,2,2)):
        features, encoder_out = inputs
        upsample = layers.UpSampling3D(size=strides)(encoder_out)
        conv_1 = self._conv_block(num_filters, upsample)
        concat = layers.Concatenate(axis=-1)([conv_1, features])
        conv_2 = self._conv_block(num_filters, concat)
        return conv_2

    def _decoder_block_last(self, num_filters,  inputs, strides=(2,2,2)):
        features, encoder_out = inputs
        upsample = layers.UpSampling3D(size=strides)(encoder_out)
        conv_1 = self._conv_block(num_filters, upsample)
        concat = layers.Concatenate(axis=-1)([conv_1, features])
        conv_2 = self._conv_block(num_filters*2, concat)
        return conv_2
    
