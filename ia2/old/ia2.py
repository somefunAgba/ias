# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

from src.infer_models import infer
from src.preprocess import preprocess
from src.archs_models import stdlogistic
from src.opts_models import batchgd
import os
import sys
import pathlib
from pathlib import Path

import numpy as np
from numpy.core.shape_base import block
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Sans Serif"]})  # Avant Garde, Helvetica, Computer Modern Sans Serif
# for Palatino and other serif fonts use:
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"], # Times, Bookman, Pa;atino
# })
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "monospace",
#     "font.monospace": ["Consolas"],
# })

np.set_printoptions(precision=4)
np.set_printoptions(formatter={'float': "{:0.4f}".format})



# %%
# Debug use

# Ensure path is referenced to this script's root
thisdir = os.path.dirname(__file__)
# os.chdir(thisdir)
os.chdir(sys.path[0])
print(os.getcwd())

figs_dir = os.path.join(thisdir, 'figs/')
if not os.path.isdir(figs_dir):
    os.makedirs(figs_dir)

# os.chdir(r'./ai534ias/ia1/')

# Generate the path to the file relative to your python script:
# script_location = Path(__file__).absolute().parent
# print(script_location)
# file_location = script_location / 'file.yaml'
# file = file_location.open()


# %%
# Data Preprocessing

# do major feature engineering - 0 | 1
doengr = 0

# Dev

rawdata = 'csvs/IA2-train.csv'
donormalize = 1

traindata, train_id = preprocess(rawdata, donormalize=donormalize, istrain=1, 
                            traininfo=None, doengr=doengr)
# View final data entering the model.
# print(traindata['X'])

rawdata = 'csvs/IA2-dev.csv'
devdata, dev_id = preprocess(rawdata, donormalize=donormalize, istrain=0, 
                            traininfo=traindata, doengr=doengr)


# %%
# DEV: Model Training and Selection

# - max. number of iterations (fixed) - epochs
epochs = int(5e3)

# - learning-rate (step-size) selection set
# lrs =  [5e-3, 1e-2, 2e-2, 0.1, 0.5]

# - regularization scale size selection set
lregs = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
# lregs = [1e-3, 1e-2, 1e-1, 1]
# regsize = 0.01 # 1e-1 to 1e-2 to 1e-3

# - random weight initialization
W = np.random.uniform(0, 0.02, (traindata['cols'], 1))
W[0] = np.zeros(shape=(1,1)) 

# Turns out lists and dicts are passed by ref. in python.
# They behave as global variables. modified in function they are passed to.

# model's number of largest weighted features
bigks = 5
# learning rate
stepsize = 1e-1
# list to hold all models
model_sels = []

# regularization rate
regtype = 1
regtype = 2
# regtype = 3

# for stepsize in lrs:
for regsize in lregs:

    print(f'\n*******L{regtype}: Model Selection************************')
    print('(start): Regularization-scale: ', regsize)
    # print(W.T) # to debug muatbility

    # modeldict is the data structure that holds details of the trained model
    modeldict = {'W': W.copy(), 'stepsize': stepsize,
                 'reg_type': regtype, 'reg_size': regsize, 'epochs': epochs,
                 'cols': traindata['cols'],
                 'normalize': traindata['scalers'],
                 'mse_train': None, 'mse_dev': None,
                 'facc_train': None, 'facc_dev': None,
                 'sparsity': None, 'bigk': bigks, 'WBigK': None, 'WBigK_feats': None
                 }

    # train: iterative line search
    batchgd(modeldict, stdlogistic, traindata, devdata)

    model_sels.append(modeldict)

    if np.isfinite(modeldict['mse_train'][-1]):
        print(f"MSE (Train): {modeldict['mse_train'][-1]:2.4f} | "
              f"(Validation): {modeldict['mse_dev'][-1]:2.4f}")
        print(f"Class Accuracy (Train): {modeldict['facc_train'][-1]:2.4f} | "
              f"(Validation): {modeldict['facc_dev'][-1]:2.4f}")
        # fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
        # ax.plot(modeldict['mse_train'], color='r', label='train')
        # ax.plot(modeldict['mse_dev'], color='r',
        #         label='validation', linestyle='--')
        # ax.set_xlabel('epochs')
        # ax.set_ylabel('cost (MSE)', color='r')
        # ax.set_title(f"$\\mathcal{{L}}_{{{regtype:1d}}}$, $\\alpha = {stepsize:2.1g}$, $\\lambda = {regsize:2.1g}$",
        #              color='k', weight='bold', size=10)
        # ax2 = ax.twinx()
        # ax2.plot(modeldict['facc_train'], color='g', label='train')
        # ax2.plot(modeldict['facc_dev'], color='g',
        #          label='validation', linestyle='--')
        # ax2.set_ylabel('class accuracy', color='g')
        # # locx = lower,upper,center, locy = left,right,center
        # ax.legend(fontsize=10, loc='center left')
        # ax2.legend(fontsize=10, loc='center right')
        # # plt.ion
        # plt.show(block=False)
        # plt.savefig(figs_dir + f"L{regtype}_{regsize:2.4f}.pdf", bbox_inches='tight')
        # # plt.close(fig)

        # print(W.T)
        print(f"Model Sparsity: {modeldict['sparsity']}")
        print('Final Learned Weights')
        # print((modeldict['W']).T)
        # Print top 5 weighted features
        print("Features (Top-5) with largest weight magnitude")
        print(modeldict['WBigK_feats'].tolist())
        wbigk_str = np.array2string(modeldict['WBigK'].flatten(), 
            formatter={'float_kind':'{0:2.4f}'.format})
        print(wbigk_str)
        print('(end): ----\n')


# Save model with least val mse.
# if more than 2, take the one with lowest learn rate.
# find weights that are non-zero (>0.1) and index
# return sparsity ratio


faccs_train = []
faccs_dev = []
regular_szs = []
sparse_wts = []
for mdl in model_sels:
    faccs_train.append(mdl['facc_train'][-1])
    faccs_dev.append(mdl['facc_dev'][-1])
    sparse_wts.append(mdl['sparsity'])
    regular_szs.append(mdl['reg_size'])

print('Regularization Plot...\t')
fig, ax3 = plt.subplots(figsize=(6, 4), tight_layout=True)
ax3.semilogx(regular_szs, faccs_train, 
    color='r', marker='o', markerfacecolor='m', zorder = 2.5, alpha=0.5)
ax3.set_ylabel(f'training accuracy', color='r')
ax4 = ax3.twinx()
ax4.semilogx(regular_szs, faccs_dev, 
    color='b', marker='o', markerfacecolor='m', zorder = 2.5, alpha=0.5)
ax4.set_ylabel(f'validation accuracy', color='b')
ax3.set_xlabel(f'$\\lambda$, regularization size')
ax3.set_title(f"$\\mathcal{{L}}_{{{regtype:1d}}}$, $\\alpha = {stepsize:2.1g}$, epochs = {epochs}: Classification Accuracy",
              color='k', weight='bold', size=10)
# plt.ion
plt.show(block=False)
plt.savefig(figs_dir + f"L{regtype}_trainvalcmp_plt.pdf", bbox_inches='tight')
print('Done.\n')


print('Sparsity Plot...\t')

fig, ax5 = plt.subplots(figsize=(6, 4), tight_layout=True)
ax5.semilogx(regular_szs, sparse_wts, 
    color='m', marker='o', markerfacecolor='r', zorder = 2.5, alpha=0.5)
ax5.set_xlabel(f'$\\lambda$, regularization size')
ax5.set_ylabel(f'weight sparsity')
ax5.set_title(f"$\\mathcal{{L}}_{{{regtype:1d}}}$, Model Sparsity",
              color='k', weight='bold', size=10)
# plt.ion
plt.show(block=False)
plt.savefig(figs_dir + f"L{regtype}_sparsityplt.pdf", bbox_inches='tight')
print('Done.\n')
