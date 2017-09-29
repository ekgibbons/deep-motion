from __future__ import print_function

import glob
import os

import numpy as np
from scipy import io
from matplotlib import pyplot as plt

def BatchGenerator(data,batchSize,iteration):
    """
    """

    firstFrameBatch = np.zeros(shape=(batchSize,144,144),dtype=np.float32)
    middleFrameBatch = np.zeros(shape=(batchSize,144,144),dtype=np.float32)
    lastFrameBatch = np.zeros(shape=(batchSize,144,144),dtype=np.float32)

    for ii in range(batchSize/2):
        firstFrameBatch[ii] = data[:,:,0,ii]
        middleFrameBatch[ii] = data[:,:,1,ii]
        lastFrameBatch[ii] = data[:,:,2,ii]

        firstFrameBatch[ii+batchSize/2] = data[:,:,2,ii]
        middleFrameBatch[ii+batchSize/2] = data[:,:,3,ii]
        lastFrameBatch[ii+batchSize/2] = data[:,:,4,ii]

    firstFrameBatch = firstFrameBatch[:,:,:,np.newaxis]
    middleFrameBatch = middleFrameBatch[:,:,:,np.newaxis]
    lastFrameBatch = lastFrameBatch[:,:,:,np.newaxis]

    y = middleFrameBatch
    X = np.concatenate((firstFrameBatch,lastFrameBatch),axis=3)
    
    XTemp0 = data[:,:,0,50]
    yTemp0 = data[:,:,1,50]
    XTemp1 = data[:,:,2,50]

    im = np.concatenate((XTemp0,yTemp0,XTemp1),axis=1)
    
    plt.figure()
    plt.imshow(abs(im))

    print(X.shape)

    XTemp0 = X[50,:,:,0].squeeze()
    XTemp1 = X[50,:,:,1].squeeze()
    yTemp0 = y[50,:,:,0].squeeze()

    im = np.concatenate((XTemp0,yTemp0,XTemp1),axis=1)

    plt.figure()
    plt.imshow(abs(im))
    plt.show()

    return X, y

def main():
    """
    """

    pathRead = "/v/raid1a/egibbons/data/deep-slice"
    data = np.load("%s/training_hearts_00.npy" % pathRead)

    print(data.shape)

    batchSize = 100
    maxBatches = 10000/batchSize

    for ii in range(maxBatches):
        X, y = BatchGenerator(data,batchSize,ii)



if __name__ == "__main__":
    main()
