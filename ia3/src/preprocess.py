# somefuno@oregonstate.edu
# Read and Preprocess CSV data: (train/dev/test)

import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# expects:

# raw data
# normalize
# is it train data.
# train data statistics
# skip feature enigneering

# normalize: age, annual premium and vintage

# outputs
# processed raw data

# Read and Preprocess CSV data: (train/dev/test)
def preprocess(rawdata, donormalize=0, istrain=1, traininfo=None, doengr=0):
    dataframe = pd.read_csv(rawdata)
    outfeature = 'Response'

    print('data size (rows,columns)', dataframe.shape)
    # print(dataframe.head())
    # print(dataframe.describe())
    data_id = dataframe.index

    dfcpy = dataframe.copy()
    # print(dataframe.columns)

    # bigwtcols = ['Previously_Insured', 'Vehicle_Damage', 'dummy']

    # Normalize only selected: numerical features to normal dists.
    normfeats = ['Age', 'Annual_Premium', 'Vintage']
    logxfmfeats = ['Annual_Premium']
    numels = len(normfeats)
    # Alternatively, to normalize all features, use
    allfeats = list(dfcpy.columns)
    if len(intersect([outfeature],allfeats)) == 1: 
        allfeats = allfeats[:-1]    
        #print(dfcpy[bigwtcols].corr())
        print(dfcpy[['Vehicle_Damage','Previously_Insured','Vehicle_Age_0', 'Vehicle_Age_1', 'Vehicle_Age_2','Response']].corr())

    #  Optional: Extra Feature Engineering
    if doengr == 1:
        corrmat = dataframe.corr()
        try:
            print('')
            # print(corrmat[outfeature], end='\n\n')
        except KeyError as ke:
            print('Key error [Response]: No ' +
                  outfeature+'-column in test-data')

        # ---- Extra Feature Eng. (start)
        # - Log transform, done before normalization
        # copy frame to operate on it.
        cpy = dfcpy.copy()
        for selfeats in logxfmfeats:
            xvals = np.sqrt(np.log(dfcpy[selfeats].values))
            cpy.loc[:, selfeats] = xvals
        dfcpy = cpy

        # - fix outliers
        # - replace with quantile Values : IQR proximity rule.

        # dataframe.columns
        for feats in normfeats:
            q1 = dfcpy[feats].quantile(0.25)
            q3 = dfcpy[feats].quantile(0.75)
            iqr = q3-q1
            extlow = q1 - (1.5*iqr)
            exthigh = q3 + (1.5*iqr)

            # selecting and indexing by masking, inplace should be true so as to set the change
            # dataframe[feats].mask(dataframe[feats] > exthigh, exthigh, inplace=True)
            # dataframe[feats].mask(dataframe[feats] < extlow, extlow, inplace=True)
            # OR if inplace == false, do as below, to eliminate pandas view-copy warning:
            cpy = dfcpy[feats].mask(dfcpy[feats] > exthigh, exthigh)
            dataframe[feats] = cpy
            cpy = dfcpy[feats].mask(dfcpy[feats] < extlow, extlow)
            dataframe[feats] = cpy

        # drop features
        dropfeats = [
            'Gender',
            'Age',
            'Annual_Premium',
            'Vintage', 
            #'Vehicle_Age_2'
        ]
        dfcpy = dfcpy.drop(columns=dropfeats)
        allfeats = differ(allfeats,dropfeats)

        # dropfeats = differ(allfeats,bigwtcols)
        # dfcpy = dfcpy.drop(columns=dropfeats)
        # allfeats = differ(allfeats,dropfeats)
        # ---- Extra Feature Eng. (end)


    # account for dropped features, if any
    allfeats = list(dfcpy.columns)
    allfeats = allfeats[:-1]
    normfeats = intersect(normfeats, allfeats)
    numels = len(normfeats)
    if (donormalize == 1) and (numels > 0):

        # apply z-score normalization
        if istrain:

            mean_data = []
            std_data = []
            for selfeats in normfeats:
                means = dfcpy[selfeats].mean()
                stds = dfcpy[selfeats].std()
                mean_data.append(means)
                std_data.append(stds)

            # keep scalers for future validation/testing
            # reshape as rows for pd.Dataframe
            mean_data = np.asarray(mean_data).reshape(1, numels)
            std_data = np.asarray(std_data).reshape(1, numels)
            # keep
            dfx_mean = pd.DataFrame(mean_data, columns=normfeats)
            dfx_std = pd.DataFrame(std_data, columns=normfeats)

            # naive: Xmean = np.zeros((dfcpy.shape[1], 1), dtype=float)
            # Xstd = np.zeros((dfcpy.shape[1], 1), dtype=float)

            # get scalers for future validation/testing
            # for id in range(Xmean.shape[0]):
            #     Xmean[id] = dfcpy.iloc[:, id].mean()
            #     Xstd[id] = dfcpy.iloc[:, id].std()
        else:
            # normalize validation/test datasets with training set's stats
            dfx_mean = traininfo['scalers']['mean']
            dfx_std = traininfo['scalers']['std']
            # print('')

        scalers = {"mean": dfx_mean, "std": dfx_std}

        # copy frame to operate on it.
        cpy = dfcpy.copy()
        for selfeats in normfeats:
            xvals = (dfcpy[selfeats].values -
                     dfx_mean[selfeats].values) / dfx_std[selfeats].values
            cpy.loc[:, selfeats] = xvals
        dfcpy = cpy

        # naive: normalize all except certain features
        # for xcols in dfcpy:
        #     id = dfcpy.columns.get_loc(xcols)
        #     if (xcols != 'waterfront') and (xcols != 'bias'):
        #         dfcpy.iloc[:,id] = (dfcpy.iloc[:,id] - Xmean[id])/(Xstd[id])

    # Extract X-Y parts
    try:
        Yout = dfcpy[outfeature].to_numpy().reshape(dfcpy.shape[0], 1)
        # 0/1 -> -1/1
        Yout = np.where(Yout == 0, -1, 1)
    except KeyError as ke:
        print('Key error [price]: No price-column in test-data')
        Yout = []

    if len(Yout) != 0:
        Xin = dfcpy.iloc[:, 0:dfcpy.shape[1]-1]
    else:
        # for test-data where output feature is not included in the raw dataset
        Xin = dfcpy.iloc[:, 0:dfcpy.shape[1]]

    # for the select fetures: make a copy of the names
    feats_name = list(Xin.columns)
    Xout = Xin.to_numpy().reshape(Xin.shape[0], Xin.shape[1])
    if len(Yout) == 0:
        print(feats_name)

    if (donormalize==1) and (numels > 0):
        indata = {'X': Xout, 'Y': Yout,
                  'rows': Xin.shape[0], 'cols': Xin.shape[1],
                  'scalers': scalers, 'feats': feats_name}
    else:
        indata = {'X': Xout, 'Y': Yout,
                  'rows': Xin.shape[0], 'cols': Xin.shape[1],
                  'scalers': None, 'feats': feats_name}

    return indata, data_id


# Helper Functions

def unique(a):
    # returns the unique list of an iterable object
    return list(set(a))


def intersect(a, b):
    # returns the intersection list of two iterable objects
    return list(set(a) & set(b))


def union(a, b):
    # returns the union list of two iterable objects
    return list(set(a) | set(b))


def differ(a, b):
    # returns the list of elements in the left iterable object
    # not in the right iterable object
    return list(set(a) - set(b))
