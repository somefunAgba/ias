# somefuno@oregonstate.edu
# Gradient-Based Optimization or Learning API


from src.infer_models import infer
from src.archs_models import *
import numpy as np
import pandas as pd
np.set_printoptions(precision=4)


# expects:
# modeldictionary,
# model,
# traindata,
# devdata

# returns:
# mse train, and  mse validation
# model dictionary


def batchgd(modeldict, model, traindata, devdata):
    mse = []
    faccs = []
    paccs = []

    mse_val = []
    facc_val = []
    pacc_val = []

    # extract
    batchlen = traindata["rows"]
    X = traindata["X"]
    Y = traindata["Y"]
    W = modeldict["W"]
    lambda_lr = modeldict["stepsize"]
    regula = modeldict["reg_type"]
    lambda_reg = modeldict["reg_size"]
    epochs = modeldict["epochs"]

    # no change counter
    cntn = 0
    # continuous divergence counter
    cntd = cntn
    for k in range(epochs):

        # shuffle data
        # shuffled_ids = np.random.permutation(batchlen)
        # X = X[shuffled_ids, :]
        # Y = Y[shuffled_ids, :]

        # predict
        Yhat = model(X, W)
        # error
        e = Y - Yhat

        # decision boundary ~ 0.5
        # y_hat(y_hat >= 0.5) = 1;
        # y_hat(y_hat < 0.5) = 0;
        decbnds = 0.5
        Yhat = np.where(Yhat > decbnds, 1, 0)
        # error
        # e = Y - Yhat

        # accuracy
        num_correct = np.sum(Yhat == Y)
        facc = num_correct/batchlen
        pacc = facc*100

        # mse: cost
        error_cost = np.sum(np.square(e))/batchlen
        mse.append(error_cost)
        faccs.append(facc)
        paccs.append(pacc)

        # early stop
        # convergence or floating-point errors
        if (mse[k] <= 1e-4) or (not np.isfinite(mse[k])):
            if not np.isfinite(mse[k]):
                print('Terminate: mse -> inf')
            break
        # no change, or divergence
        if k > 0:
            if mse[k] == mse[k - 1]:
                cntn += 1
                if cntn > 50:
                    print('Stop: mse -> no change, static', k)
                    break
            if mse[k] > 20 * mse[k - 1]:
                cntd += 1
                if cntd > 25:
                    print('Stop: mse -> no change, increasing', k)
                    break

        # GD: Batch

        # output jacobian
        J = X

        # first-order gradient of the cost-function
        # g = np.dot(J.T,e) #
        # g = np.einsum("ij,ik->jk", J, e)
        g = (J.T)@(e)

        # step size multiplied with descent direction
        p = (lambda_lr/batchlen)*(np.pi*g)
        # print(np.size(p))

        # weight change
        dW = p

        # - update
        W += dW
        # - regularize, de-regularize bias
        if regula == 2:
            W -=(lambda_lr*lambda_reg)*W
            W[0] +=(lambda_lr*lambda_reg)*W[0]
        elif regula == 1:
            W -= (lambda_lr*lambda_reg)*np.sign(W)
            W[0] += (lambda_lr*lambda_reg)*np.sign(W[0])
        elif regula == 3:
            aw = 0.2 # [0 1]
            elasticW = (1-aw)*np.sign(W) + (aw*W)
            W -= (lambda_lr*lambda_reg)*(elasticW)
            W[0] += (lambda_lr*lambda_reg)*elasticW[0]
        elif (regula != 2) or (regula != 1) or (regula != 3):
            W = W
            # do nothing

        

        # optional - regularized weight update
        # - L2 norm (ridge)
        # - L1 norm (lasso)
        # - excluding bias weight: i.e: feature 1 to end, exclude 0th index
        # if regula == 2:
        #     W[1:] -= (lambda_lr*lambda_reg)*W[1:] 
        # elif regula == 1:
        #     W[1:] -= (lambda_lr*lambda_reg)*np.sign(W[1:])
        # elif regula == 3:
        #     W[1:] -= (lambda_lr*lambda_reg)*(W[1:] + np.sign(W[1:]))


        modeldict["W"] = W
        [Yhatv, msev, faccv, paccv] = infer(model, modeldict, devdata)
        mse_val.append(msev)
        facc_val.append(faccv)
        pacc_val.append(paccv)


        # log progress
        if k % 500 == 0:
            messagelog = (
                f"k: {k:5d}, mse(train): {mse[k]:2.4f}, mse(dev): {mse_val[k]:2.4f} | "
                f"facc(train): {faccs[k]:2.4f}, facc(dev): {facc_val[k]:2.4f}"
            )
            print(messagelog)
    
    
    # - compute weight sparsity for this model
    sparse_num = np.sum(W <= 1e-6)
    # print(f"Model Sparsity: {sparse_num}")
    # store
    modeldict['sparsity'] = sparse_num

    # - find top 5 features with the largest magnitude
    # sort in descending order
    feats_idx = (-np.abs(W)).argsort(axis=0).T
    bigk = modeldict['bigk'] 
    # get big 5 or big k  ids as may be the case
    feats_idx_bigK = feats_idx[0,:bigk]
    # get corresponding features
    # feats_bigKwts = [traindata['feats'][idx] for idx in feats_idx_bigK]
    feats_bigKwts = np.array(traindata["feats"])[feats_idx_bigK]
    WBigK = W[feats_idx_bigK].T
    WBigK_feats = feats_bigKwts
    # print(f"(Big-5) Learned Weights: {WBigK}")
    # print(f"(Big-5) Corresponding Features: {WBigK_feats}")
    # store
    modeldict["WBigK"] = WBigK
    modeldict['WBigK_feats'] = WBigK_feats

    # store model params
    modeldict["W"] = W

    # - store metrics
    modeldict['mse_train'] = mse
    modeldict['mse_dev'] = mse_val
    modeldict['facc_train'] = faccs
    modeldict['facc_dev'] = facc_val

    

