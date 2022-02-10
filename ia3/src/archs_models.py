# somefuno@oregonstate.edu
# Architectures, Models, Hypothesis Spaces API

import numpy as np

# - Linear Model: linear(X,W)
# expects: 
# X.shape: i x j,
# W.shape = j x k

# returns: Y.shape: i x k

def linear(X,W):
    # return X.dot(W) # 
    # return np.einsum('ij,jk->ik',X,W)
    return X@W

# Xt: input training data: N x d, N>=1
# X: input data:  Q x d, Q>=1
# p: polynomial degree / order, p>=1
# Yt: output training data: N x 1, N>=1
# alphat: kernel weight:  N x 1, N>=1
def polykernel(Xt,X,p):
    return np.power(X@(Xt.T),p)


def linear_kernelized(Xt,Yt,alphat,p,X):
    return (polykernel(Xt,X,p))@((alphat*Yt))


def sgnlinear_kernelized(Xt,Yt,alphat,p,X):
    v = linear_kernelized(Xt,Yt,alphat,p,X)
    return np.sign(v)



# - Sign Linear Model: sgnlinear(X,W)
# expects: 
# X.shape: i x j,
# W.shape = j x k

# returns: Y.shape: i x k

def sgnlinear(X,W):
    v = linear(X,W)
    return np.sign(v)



# - Standard Logistic Model: stdlogistic(X,W)
# expects: 
# X.shape: i x j,
# W.shape = j x k

# returns: Y.shape: i x k


def stdlogistic(X,W):
    # v = X.dot(W) 
    # v = np.einsum('ij,jk->ik',X,W)
    v = linear(X,W)
    return  1/(1 + np.exp(-v))



