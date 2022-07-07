import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import models

from math import pi
from math import cos
from math import floor

class SaveModelOnCD(callbacks.Callback):
    def __init__(self, keys, model_save_path, patience, grid_weight=None, grid_key=None):
        self.keys = keys
        self.save_path = model_save_path
        self.no_improve = 0
        self.patience = patience
        self.grid_weight = grid_weight
        self.grid_key = grid_key
    
    def on_train_begin(self, logs=None):
        self.best = np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        CD_val_loss = 0.
        for key in self.keys:
            CD_val_loss += logs.get('val_'+key+'_point_loss_cf')
        if self.grid_weight is not None:
            mean_CD_loss = CD_val_loss/float(len(self.keys))
            grid_loss = logs.get('val_'+self.grid_key+'_loss')
            new_weight = min(mean_CD_loss / grid_loss * K.get_value(self.grid_weight), 1000)
            K.set_value(self.grid_weight, new_weight)
            print("Setting grid loss {} weight to: {}.".format('val_'+self.grid_key+'_loss', K.get_value(self.grid_weight) ))
        if CD_val_loss < self.best:
            path = self.save_path.format(epoch)
            print("Epoch: ", epoch, path)
            print("\nValidation loss decreased from %f to %f, saving model to %s.\n" % (self.best, CD_val_loss, path))
            self.best = CD_val_loss
            if isinstance(self.model.layers[-2], models.Model):
                self.model.layers[-2].save_weights(path)
            else:
                self.model.save_weights(path)
            self.no_improve = 0
        else:
            print("\nValidation loss did not improve from %f.\n" % self.best)
            self.no_improve += 1
        if self.no_improve > self.patience:
            self.model.stop_training = True

class ReduceLossWeight(callbacks.Callback):
    def __init__(self, grid_weight, patience=10, factor=0.5):
        self.grid_weight = grid_weight
        self.num_epochs = 0
        self.patience = patience
        self.factor = factor
    
    def on_epoch_end(self, epoch, logs=None):
        self.num_epochs += 1
        if self.num_epochs >= self.patience:
            new_weight = max(K.get_value(self.grid_weight) * self.factor, 10.)
            K.set_value(self.grid_weight, new_weight)
            print("Setting grid loss weight to: {}.".format(K.get_value(self.grid_weight)))
            self.num_epochs = 0

