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
    X = indata["X"].copy()
    Y = indata["Y"].copy()

    usekernel = modeldict["usekernel"]

    # predict
    if usekernel == 0:
        W = modeldict["W"].copy()
        Yhat = model(X, W)
    else:
        Xt = modeldict["Xt"].copy()
        Yt = modeldict["Yt"].copy()
        alphat = modeldict["alphat"].copy()
        p = modeldict["p"]
        X = indata["X"].copy()
        Yhat = model(Xt, Yt, alphat, p, X)
        

    # metrics: mean-squared error, classification accuracy
    if len(Y) != 0:
        e = Y - Yhat
        # mse: cost
        mse = np.sum(np.square(e))/batchlen
        # np.divide(np.square(np.linalg.norm(e)), batchlen)

        if modeldict['bndtype'] == 0.5:
            decbnds = 0.5
            Yhat = np.where(Yhat > decbnds, 1, 0)
        elif modeldict['bndtype'] == 0:
            decbnds = 0
            Yhat = np.where(Yhat > decbnds, 1, -1)

        # 0/1 Accuracy
        num_correct = np.sum(Yhat == Y)
        facc = num_correct/batchlen
        pacc = facc*100
        return [Yhat, mse, facc, pacc]
    else:
        if modeldict['bndtype'] == 0.5:
            decbnds = 0.5
            Yhat = np.where(Yhat > decbnds, 1, 0)
        elif modeldict['bndtype'] == 0:
            decbnds = 0
            Yhat = np.where(Yhat > decbnds, 1, -1)
        return Yhat
