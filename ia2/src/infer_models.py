# somefuno@oregonstate.edu
# Inference API

import numpy as np


# expects:
# model: machine learning hypothesis space or algorithm
# modeldict: dictionary data structure containing model parameters
# indata: input data

# outputs:
# Yhat: predicted output
# mse: cost, if supervised
def infer(model, modeldict, indata):
    mse = []

    # extract
    batchlen = indata["rows"]
    X = indata["X"]
    Y = indata["Y"]
    W = modeldict["W"]

    # predict
    Yhat = model(X, W)

    # metrics: mean-squared error, classification accuracy
    if len(Y) != 0:
        e = Y - Yhat
        # mse: cost
        mse = np.sum(np.square(e))/batchlen
        # np.divide(np.square(np.linalg.norm(e)), batchlen)
        
        decbnds = 0.5
        Yhat = np.where(Yhat > decbnds, 1, 0)

        # accuracy
        num_correct = np.sum(Yhat == Y)
        facc = num_correct/batchlen
        pacc = facc*100
        return [Yhat, mse, facc, pacc]
    else:
        decbnds = 0.5
        Yhat = np.where(Yhat > decbnds, 1, 0)
        return Yhat
