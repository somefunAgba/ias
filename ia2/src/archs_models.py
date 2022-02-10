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
    return np.einsum('ij,jk->ik',X,W);


# - Standard Logistic Model: stdlogistic(X,W)
# expects: 
# X.shape: i x j,
# W.shape = j x k

# returns: Y.shape: i x k


def stdlogistic(X,W):
    # v = X.dot(W) 
    # v = np.einsum('ij,jk->ik',X,W)
    v = X@W
    return  1/(1 + np.exp(-v))



