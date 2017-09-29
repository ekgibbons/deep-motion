# Script to train a CNN to do image interpolation using Keras
# After experimentation, I've settled on the following network architecture and training scheme:
#  - Using a modified version of U-Net
#       (because of the addition of batch normalization layers, the network inputs must be a fixed shape,
#       for some given model weights. Kind of defeats the purpose of using a fully-convolutional network,
#       but I ran out of time. Maybe in the future I will improve the architecture,
#       and remove batch norm so variable sized inputs work.)
#
#       In any case, the provided weights:
#           "weights_unet2_finetune_youtube_100epochs.hdf5"
#       are trained with this architecture, so the input must be 6x128x384
#  - Adam optimizer with a learning rate of 0.0001 (dynamic), batch size of 16
#  - Optimizing the charbonnier loss function
#  - Training on the KITTI dataset at first, then finetuning on YouTube-8M


import sys
import os

# BACKEND = "theano"
BACKEND = "tensorflow"

os.environ['KERAS_BACKEND'] = BACKEND
os.environ['THEANO_FLAGS'] = "device=gpu0, lib.cnmem=0.85, optimizer=fast_run"

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

LOAD_PRE_TRAINED_MODEL = True
DO_TESTING = True

# need to use this when I require batches from an already memory-loaded X, y
def np_array_batch_generator(X, y, batch_size):
    batch_i = 0
    while 1:
        if (batch_i+1)*batch_size >= len(X):
            yield X[batch_i*batch_size:], y[batch_i*batch_size:]
            batch_i = 0
        else:
            yield X[batch_i*batch_size:(batch_i+1)*batch_size], y[batch_i*batch_size:(batch_i+1)*batch_size]


def main():
    ##### DATA SETUP #####
    # the only pre-processing is to divide by 255, to make pixel values between 0 and 1

    X_val = np.load("X_val_KITTI.npy").astype("float32") / 255.
    y_val = np.load("y_val_KITTI.npy").astype("float32") / 255.

    ##### MODEL SETUP #####
    model = get_unet_2(input_shape=(2, 144, 144))

    optimizer = adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    loss = charbonnier

    model.compile(loss=loss, optimizer=optimizer)

    if DO_TESTING:
        # Do predictions using weights for network trained on just the KITTI dataset
        model.load_weights("./../model_weights/weights_kitti_167plus25epochs_unet2_ch_pt03136_best_but_fixed_imsize.hdf5")
        X, y = kitti_batch_generator(50).next()
        y_pred = model.predict(X, batch_size=BATCH_SIZE, verbose=1)

        # Do second predictions on more refined weights, finetuned on YouTube-8M
        model.load_weights("./../model_weights/weights_unet2_finetune_youtube_100epochs.hdf5")

        y_pred_2 = model.predict(X, batch_size=BATCH_SIZE, verbose=1)

        return 0

    ##### TRAINING SETUP #####
    logging.warning("USING loss AS MODEL CHECKPOINT METRIC, CHANGE LATER!")
    callbacks = [
        ModelCheckpoint(filepath="./../model_weights/weights.hdf5", monitor='loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, verbose=1)
    ]
    # callbacks.append(TensorBoard(log_dir="./../tensorboard_logs", write_graph=False))

    if LOAD_PRE_TRAINED_MODEL:
        print ""
        logging.warning("LOADING PRE-TRAINED MODEL WEIGHTS!")
        print ""
        model.load_weights("./../weights.hdf5")
        callbacks.append(CSVLogger("stats_per_epoch.csv", append=True))
    else:
        callbacks.append(CSVLogger("stats_per_epoch.csv", append=False))


    # OPTION 1: train on batches from youtube-8m
    BATCH_IMAGE_SIZE = (144, 144)
    print "Begin training..."
    hist = model.fit_generator(
        generator=batch_generator(BATCH_SIZE, NUM_CHANNELS, BATCH_IMAGE_SIZE),
        samples_per_epoch=800,
        nb_epoch=NUM_EPOCHS,
        callbacks=callbacks,
        validation_data=(X_val, y_val),
        max_q_size=10,
        # nb_worker=1,
        nb_worker=cpu_count(),
    )

if __name__ == '__main__':
    main()
