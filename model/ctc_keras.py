import argparse
import numpy as np
import cv2
import os
import itertools
import tensorflow as tf

from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, BatchNormalization, LeakyReLU
from keras.layers import Reshape, Lambda, Permute
from keras.models import Model
from keras.layers.recurrent import GRU, LSTM
from keras.layers import TimeDistributed, Bidirectional
from keras.backend.tensorflow_backend import set_session


# ===================================================



def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# TODO Editar el modelo para que sea como el de tensorflow
def get_model(input_shape, vocabulary_size):
    
    init_session()
    
    conv_filters = [16,32,64]

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    inner = Conv2D(conv_filters[0], 3, padding='same', name='conv1')(input_data)
    inner = BatchNormalization()(inner)
    inner = LeakyReLU(alpha=0.2)(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

    inner = Conv2D(conv_filters[1], 3, padding='same', name='conv2')(inner)
    inner = BatchNormalization()(inner)
    inner = LeakyReLU(alpha=0.2)(inner)
    inner = MaxPooling2D(pool_size=(2, 1), name='max2')(inner)

    inner = Conv2D(conv_filters[2], 3, padding='same')(inner)
    inner = BatchNormalization()(inner)
    inner = LeakyReLU(alpha=0.2)(inner)
    inner = MaxPooling2D(pool_size=(2, 1))(inner)

    inner = Permute((2, 1, 3))(inner)
    inner = Reshape(target_shape=(-1, (input_shape[0] // (2 ** 3)) * conv_filters[-1]), name='reshape')(inner)

    inner = Bidirectional(LSTM(64, return_sequences = True, dropout=0.25))(inner)

    inner = Dense(vocabulary_size+1, name='dense2')(inner)
    y_pred = Activation('softmax', name='softmax')(inner)

    model_pr = Model(inputs=input_data, outputs=y_pred)
    model_pr.summary()

    labels = Input(name='the_labels',shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model_tr = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=loss_out)

    model_tr.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    # Nota: el modelo que se entrena es el que lleva ctc (model_tr) pero para guardarlo usamos el que no (model_pr)
    return model_tr, model_pr

def init_session():
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth= True
    sess = tf.Session(config=conf)
    set_session(sess)