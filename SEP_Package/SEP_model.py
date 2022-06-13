'''
 (c) Copyright 2022
 All rights reserved
 Programs written by Yasser Abduallah
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA

 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.

 @author: Yasser Abduallah
'''

from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
try:  
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    log('')
import numpy as np
from datetime import datetime, timedelta
import argparse
import time
from time import sleep
import math
import random
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.callbacks import EarlyStopping
import shutil

from sklearn.utils import class_weight
from SEP_utils import *
from SEP_attention import *

class SEPModel:
    model = None
    model_name = None
    callbacks = None
    input = None
    input_shape = None
    loss = None
    adam_lr = None
    metrics = None
    
    def __init__(self,model_name='SEPModel',early_stopping_patience=3):
        self.model_name = model_name
        callbacks = [EarlyStopping(monitor='loss', patience=early_stopping_patience)]

    if tf.test.gpu_device_name() != '/device:GPU:0':
      log('WARNING: GPU device not found.')
    else:
        log('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices ) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
        
    
    def build_base_model(self,
                    input_shape):
            input = keras.Input(shape=input_shape)
            self.input = input   
            self.input_shape = input_shape
            model = layers.Dense(100, kernel_regularizer=l2(0.001))(input)
            model = layers.Dense(100, kernel_regularizer=l2(0.001))(input)
            model = layers.Bidirectional(layers.LSTM(units=400,kernel_regularizer=l2(0.001), return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(model)
            model = layers.Bidirectional(layers.LSTM(units=100,return_sequences=True,kernel_regularizer=l2(0.001)))(model)
            model = SEPAttention()(model)
            model = layers.Flatten()(model) 
            model = layers.Dense(1, activation='sigmoid',kernel_regularizer=l2(0.001))(model)
            model = layers.BatchNormalization(momentum=0.9)(model)
            self.model = model
            return model            

    def models(self):
        self.model = models.Model(self.input, self.model)
        
    def summary(self):
        self.model.summary(print_fn=print_summary_to_file)
    
    def compile(self,loss='binary_crossentropy',metrics=['accuracy'], adam_lr=0.001):
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=adam_lr),loss=loss, metrics=metrics)
        
    def fit(self,
            X_train, 
            y_train,
            X_valid=None, 
            y_valid=None,
            epochs=2,
            verbose=0,
            batch_size=2048):
        validation_data = None 
        if X_valid and y_valid:
            validation_data =[X_valid, y_valid]
        starting_time = datetime.now()
        
        history = self.model.fit(X_train, 
                       y_train, 
                       epochs=epochs, 
                       verbose=verbose, 
                       batch_size=batch_size,
                       callbacks=[keras.callbacks.ProgbarLogger()])
        total_seconds = int((datetime.now() - starting_time).total_seconds())
        per_step = total_seconds//4
        print('11/11','-', str(int((datetime.now() - starting_time).total_seconds())) + 's', '-','loss:',
              round(np.array(history.history['loss']).min(),4),'-', 'accuracy:', 
              round(np.array(history.history['accuracy']).max(),4),end=' ')
        # print('1/1 [==============================]','-', per_step, 's/step')
        return history
        
    def predict(self,X_test,verbose=1):
        predictions = self.model.predict(X_test,
                                         verbose=verbose,
                                         batch_size=len(X_test))
        return np.squeeze(predictions) 
    
    def save_weights(self,e_type='fc',time_window=12,w_dir=None):
        e_type = str(e_type).lower().replace('_s','')
        weight_dir = 'models' + os.sep + 'sep_model_' + str(e_type) + '_' + str(time_window) + 'hr'
        if w_dir is not None:
            weight_dir = w_dir +  os.sep + 'sep_model_' + str(e_type) + '_' + str(time_window) + 'hr'
        log('Saving weights from model dir:', weight_dir)
        if os.path.exists(weight_dir):
            shutil.rmtree(weight_dir)
        os.makedirs(weight_dir)
        self.model.save_weights(weight_dir + os.sep + 'model_weights')
    
    def load_weights(self,e_type='fc',time_window=12,w_dir=None):
        e_type = str(e_type).lower().replace('_s','')
        weight_dir = 'models' + os.sep + 'sep_model_' + str(e_type) + '_' + str(time_window) + 'hr'
        if w_dir is not None:
            weight_dir = w_dir +  os.sep + 'sep_model_' + str(e_type) + '_' + str(time_window) + 'hr'
        log('Loading weights from model dir:', weight_dir)
        if not os.path.exists(weight_dir):
            log( 'Error: Model weights directory does not exist:', weight_dir,verbose=True,format_logging=False)
            if not w_dir == 'default_models':
                log('Trying pre trained default models directory: default_models' ,verbose=True,format_logging=False)
                weight_dir = 'default_models' +  os.sep + 'sep_model_' + str(e_type) + '_' + str(time_window) + 'hr'
                if not os.path.exists(weight_dir):
                    log( 'Error: Model weights for default directory does not exist:', weight_dir,verbose=True,format_logging=False)
                    exit()
            else:
                exit()
        self.build_model(weight_dir + os.sep + 'model_weights')   
        if self.model == None :
            print('Error: You must train a model first before loading the weights.')
            exit()
        log('Loading weights from:', weight_dir + os.sep + 'model_weights',verbose=True)      
        self.model.load_weights(weight_dir + os.sep + 'model_weights').expect_partial()
    
    def load_model(self,input_shape=(series_len,n_features),
                   e_type='FC_S',
                   time_window=12,
                   loss='binary_crossentropy',
                   metrics=['accuracy'],
                   adam_lr=0.0001,
                   w_dir=None):
        self.input_shape = input_shape 
        self.adam_lr = adam_lr
        self.metrics = metrics 
        self.loss = loss 
        
        # self.build_base_model(input_shape)
        # self.models()
        # self.compile(loss=loss, metrics=metrics, adam_lr=adam_lr)
        e_type=str(e_type).lower()
        self.load_weights(e_type=e_type, time_window=time_window,w_dir=w_dir)
    
    def save(self,dir_name):
        os.makedirs(dir_name,  exist_ok=True)
    
    def build_model(self,w):
        log('Building model for:', w,verbose=True)
        from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        from tensorflow.python.training import py_checkpoint_reader
        reader = py_checkpoint_reader.NewCheckpointReader(w)
        maps = reader.get_variable_to_shape_map()
        b = maps['layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE'][0]
        a = maps['layer_with_weights-3/att_bias/.ATTRIBUTES/VARIABLE_VALUE'][0]
        self.input_shape = (a,b)
        self.build_base_model(self.input_shape)
        self.models()
        self.compile(loss=self.loss, 
                     metrics=self.metrics, 
                     adam_lr=self.adam_lr)
