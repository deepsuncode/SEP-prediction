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
from keras.layers import Layer
import keras.backend as K

class SEPAttention(Layer):
    def __init__(self,**kwargs):
        super(SEPAttention,self).__init__(**kwargs)
        SEPAttention.name='SEPCustomAttention'

    def build(self,input_shape): 
        """
        Matrices for creating the context vector.
        """
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(SEPAttention, self).build(input_shape)

    def call(self,x):
        """
        Function which does the computation and is passed through a softmax layer to calculate the attention probabilities and context vector. 
        """
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        """
        For Keras internal compatibility checking.
        """
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        """
        The get_config() method collects the input shape and other information about the model.
        """
        return super(Attention,self).get_config()