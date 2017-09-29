from __future__ import print_function
from __future__ import absolute_import

import sys
import os

BACKEND = "tensorflow"
os.environ['KERAS_BACKEND'] = BACKEND

import random
import logging
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

from keras.optimizers import SGD, adadelta, adagrad, adam, adamax, nadam
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau

from FI_CNN import FI_CNN_model, FI_CNN_model_BN
from FI_unet import *
from data_generator import batch_generator, kitti_batch_generator

from more_loss_fns import *

LEARNING_RATE = 0.0001
BATCH_SIZE = 16
NUM_EPOCHS = 1000
NUM_CHANNELS = 3

LOAD_PRE_TRAINED_MODEL = False
DO_TESTING = False


def main():
    """
    """

    # set up the data
    dataTrain = data[:,:,:,:int(data.shape[3]*4/5)]
    X, y = BactchGenerator(data,batchSize,ii)

    dataVal = data[:,:,:,int(data.shape[3]*4/5):]    

    # setup the model
    model = get_unet_2(input_shape=(2, 144, 144))
    optimizer = adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    loss = charbonnier

    model.compile(loss=loss, optimizer=optimizer)

    # not sure what this is.  Need to figure this out...
    logging.warning("USING loss AS MODEL CHECKPOINT METRIC, CHANGE LATER!")
    callbacks = [
        ModelCheckpoint(filepath="./../model_weights/weights.hdf5", monitor='loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, verbose=1)
    ]

    # train
    BATCH_IMAGE_SIZE = (144, 144)
    print "Begin training..."
    hist = model.fit_generator(
        generator=batch_generator(data, NUM_CHANNELS, BATCH_IMAGE_SIZE),
        samples_per_epoch=800,
        nb_epoch=NUM_EPOCHS,
        callbacks=callbacks,
        validation_data=(X_val, y_val),
        max_q_size=10,
        # nb_worker=1,
        nb_worker=cpu_count(),
    )




if __name__ == "__main__":
    main()
