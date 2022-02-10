# somefuno@oregonstate.edu
# Architectures, Models, Hypothesis Spaces API

import numpy as np

from src.treehelpers import *

# - Linear Model: linear(X,W)
# expects:
# X.shape: i x j,
# W.shape = j x k

# returns: Y.shape: i x k


def linear(X, W):
    # return X.dot(W) #
    # return np.einsum('ij,jk->ik',X,W)
    return X@W

# Xt: input training data: N x d, N>=1
# X: input data:  Q x d, Q>=1
# p: polynomial degree / order, p>=1
# Yt: output training data: N x 1, N>=1
# alphat: kernel weight:  N x 1, N>=1


def polykernel(Xt, X, p):
    return np.power(X@(Xt.T), p)


def linear_kernelized(Xt, Yt, alphat, p, X):
    return (polykernel(Xt, X, p))@((alphat*Yt))


def sgnlinear_kernelized(Xt, Yt, alphat, p, X):
    v = linear_kernelized(Xt, Yt, alphat, p, X)
    return np.sign(v)


# - Sign Linear Model: sgnlinear(X,W)
# expects:
# X.shape: i x j,
# W.shape = j x k

# returns: Y.shape: i x k

def sgnlinear(X, W):
    v = linear(X, W)
    return np.sign(v)


# - Standard Logistic Model: stdlogistic(X,W)
# expects:
# X.shape: i x j,
# W.shape = j x k

# returns: Y.shape: i x k


def stdlogistic(X, W):
    # v = X.dot(W)
    # v = np.einsum('ij,jk->ik',X,W)
    v = linear(X, W)
    return 1/(1 + np.exp(-v))


def stdlogistic(v):
    # v = X.dot(W)
    # v = np.einsum('ij,jk->ik',X,W)
    return 1/(1 + np.exp(-v))


##
# -------- Decision Trees -------------------
##

# Terminal Node
# - holds predictions of a terminal leaf node
# - which classifies data
class Leaf:
    def __init__(self, data_x, data_y, w):
        self.predictions = max_class(data_y, w)

# Decision Node
# - asks a question and splits further into two nodes
class Node:
    def __init__(self, node_dec, branch_t, branch_f, depth):
        self.node_dec = node_dec
        self.branch_t = branch_t
        self.branch_f = branch_f
        self.depth = depth
        pass

# Builds Decision Tree nodes
# - by recursion from main root till stopping conditions are reached
class DecisionTree:

    def __init__(self, metric_class=ent_class, feature_samplesize=0, max_depth=np.Infinity, min_size=3, inprints=1):
        self.metric_class = metric_class
        self.feature_samplesize = feature_samplesize
        self.max_depth = max_depth
        self.min_size = min_size
        self.depth = 0
        self.nodes = None
        self.inprints = 1
        pass

    def build(self, data_x, data_y, x_names, w=None, N=None, depth=1, colw=None):       

        # search over all features and
        # split dataset based on unique feature with best (max) I.G
        best_splitdec = search_bestsplit(
            data_x, data_y, x_names, self.metric_class, w, self.feature_samplesize, N, colw)
        # for k in best_splitdec:
        #     print(k, ":", best_splitdec[k])

        # base case:
        # if no further information gain, or
        # max. depth of tree or
        # min. size of samples is reached, then TERMINATE TREE growth
        # return a Leaf which:
        # predict the class with the most occurence

        if self.metric_class == ent_class:
            if (best_splitdec['ig'] == 0) | \
                    (depth > self.max_depth) | \
                    (len(data_y) < self.min_size):
                # return Leaf
                return Leaf(data_x, data_y, w)
        elif self.metric_class == gini_class:
            if (best_splitdec['ig'] == 1) | \
                    (depth > self.max_depth) | \
                    (len(data_y) < self.min_size):
                # return Leaf
                return Leaf(data_x, data_y, w)

        # I.G (best split) : ig, id, name, val
        ux = best_splitdec['val']
        qxid_best = best_splitdec['id']

        split_x = data_x[:, qxid_best]

        # partition ids by best split unique value -> this feature
        qat = (split_x == ux)  # true parts # >=, <=
        qaf = ~(qat)    # false parts

        # data split based on the best unique value of best split feature
        node_dec = best_splitdec
        split_tx = data_x[qat]
        split_ty = data_y[qat]
        if w is None:
            w_ty = None
            w_fy = None
        else:
            w_ty = w[qat]
            w_fy = w[qaf]
            
            
        split_fx = data_x[qaf]
        split_fy = data_y[qaf]
            
            
        # recursively build true and false branches

        if len(split_ty)*len(split_fy) == 0:
            return Leaf(data_x, data_y, w)
        
        
        self.depth += 1

        # - true branch
        branch_t = self.build(split_tx, split_ty, x_names, w_ty, N, depth+1)

        # - false branch
        branch_f = self.build(split_fx, split_fy, x_names, w_fy, N, depth+1)

        # return Node
        return Node(node_dec, branch_t, branch_f, depth)



    def print(self, nodes=None, gaps=">"):
        
        if nodes is None:
            nodes = self.nodes

        # base case:
        # if we have reached a leaf
        if isinstance(nodes, Leaf):
            print(f"{gaps}   (class) predict: {nodes.predictions}")
            return

        # print best split decision at this node
        print(f"[{nodes.depth-1}] {gaps} X[{nodes.node_dec['name']}] = {nodes.node_dec['val']}, (information gain: {nodes.node_dec['ig']:.2g}):")

        # recursively print above for each branch
        print(f"{gaps} --> True: ")
        self.print(nodes.branch_t, gaps + " ")

        print(f"{gaps} --> False: ")
        self.print(nodes.branch_f, gaps + " ")


    # one row K = 1 classification
    def classify(self, nodes, data_x_row):

        # base case
        if isinstance(nodes, Leaf):
            return nodes.predictions

        # I.G (best split) : ig, id, name, val
        ux = nodes.node_dec['val']
        qxid_best = nodes.node_dec['id']

        # partition ids by best split unique value -> this feature
        test_x = data_x_row[qxid_best]

        qa = (test_x == ux)  # true parts # >=, <=

        if qa:
            return self.classify(nodes.branch_t, data_x_row)
        else:
            return self.classify(nodes.branch_f, data_x_row)


    # full K rows classification
    def infer(self, data_x, data_y=None, inprints=1):
        pred_y = []
        for data_x_row in data_x:
            decision = self.classify(self.nodes, data_x_row)
            pred_y.append(decision)

        pred_y = np.array(pred_y).reshape((len(pred_y), 1))

        
        if not(data_y is None):
            batchlen = len(data_y)
            counts = np.sum(pred_y == data_y)
            # 0/1 accuracy (fractional)
            accs = counts/batchlen
            # 0/1 misses
            misses = batchlen - counts
            if inprints == 1:
                print(f"class accuracy: {accs:.2g}, misses: {misses}")

        return pred_y, accs, misses
    
# Builds Random Forest with Bagging: (Many Decision Trees)
class RandomForest:
    def __init__(self, numofTrees):
        self.T = numofTrees
        self.dectrees = []
        self.oobs = []
        self.oobs_mean = 0
        self.pfs = []
        pass

    # setup basic structure for tree ensemble 
    def ensemble_trees(self, data_x, data_y, x_names, metric_class=ent_class, feature_samplesize=0, max_depth=np.Infinity, min_size=3, inprints=1, depth=1):
        
        # row bagging
        bag_x, bag_y, x_oob, y_oob = bagging(data_x, data_y, True, p=None,skipbag=0)
        
        
        # decision tree with feature bagging
        dectree = DecisionTree(metric_class, feature_samplesize, max_depth, min_size,inprints)
        dectree.nodes = dectree.build(bag_x, bag_y, x_names, w=None, colw = self.pfs)
        
        # out-of-bag estimate
        if inprints == 1:
            print(f"OOB Estimate")
        y_oob_hat, acc_oob, misses_oob = dectree.infer(x_oob, y_oob, inprints)
        oob_estimate = acc_oob

        return dectree, oob_estimate
 
    
    def build(self, DX, DY, x_names, metric_class=ent_class, feature_samplesize=1, max_depth=np.Infinity, min_size=3, inprints=0):
        # prob. dist for feature resampling
        self.pfs = feat_split_dist(DX, DY, x_names, split_metric=metric_class)
        # print(self.pfs[np.argmax(self.pfs)])
        for id in np.arange(self.T):           
            # decision trees with bagging (row and column)
            dtree, oob_score = self.ensemble_trees(
                DX, DY, x_names, metric_class, feature_samplesize, max_depth, min_size, inprints=0)
            self.dectrees.append(dtree)
            self.oobs.append(oob_score)
        self.oobs_mean = np.mean(np.array(self.oobs))
        # out-of-bag estimate
        if inprints:
            print(f"Forest => OOB Mean-Estimate: {self.oobs_mean:.2g}")

    def infer(self, data_x, data_y=None, inprints=0):
        pred_y = []
        for data_x_row in data_x:
            ytemp = []
            for dtree in self.dectrees:
                decision = dtree.classify(dtree.nodes, data_x_row)
                ytemp.append(decision)
            # select most occuring prediction
            # maxy = max_class(np.array(ytemp).reshape((len(ytemp), 1)))
            maxy = max(ytemp, key=ytemp.count)
            # print(maxy==maxy) #expects: True
            
            pred_y.append(maxy)

        pred_y = np.array(pred_y).reshape((len(pred_y), 1))

        if len(data_y) == 0:
            return pred_y
        
        batchlen = len(data_y)
        if batchlen != 0:
            counts = np.sum(pred_y == data_y)
            # 0/1 accuracy (fractional)
            accs = counts/batchlen
            # 0/1 misses
            misses = batchlen - counts
            if inprints:
                print(f"Forest => class accuracy: {accs:.2g}, misses: {misses}")

        return pred_y, accs, misses
  
# Builds AdaBoost: Boosted Forests: (Many Decision Trees)
class AdaBoost:
    def __init__(self, numofTrees):
        self.T = numofTrees
        self.dectrees = []
        self.oobs = []
        self.oobs_mean = 0
        self.alphas = []
        self.N = None
        pass
    
    
    # setup basic structure for tree ensemble 
    def ensemble_trees(self, data_x, data_y, x_names, w, metric_class=ent_class, feature_samplesize=0, max_depth=np.Infinity, min_size=3, inprints=1, depth=1):
        
        # bagging
        bag_x, bag_y, x_oob, y_oob = bagging(data_x, data_y, True, p=None, skipbag=1)
        
        
        # decision tree
        dectree = DecisionTree(metric_class, feature_samplesize, max_depth, min_size,inprints)
        dectree.nodes = dectree.build(bag_x, bag_y, x_names, w, self.N)
        
        # out-of-bag estimate
        if inprints == 1:
            print(f"OOB Estimate")
        y_oob_hat, acc_oob, misses_oob = dectree.infer(x_oob, y_oob, inprints)
        oob_estimate = acc_oob

        y_bag_hat, acc_bag, misses_bag = dectree.infer(bag_x, bag_y, inprints)

        return dectree, y_bag_hat, bag_y, oob_estimate
            
    
    
    # build boosted ensemble
    def build_ensemble(self, DX, DY, x_names, metric_class=ent_class, feature_samplesize=1, max_depth=np.Infinity, min_size=3, inprints=0):
        w = np.ones((DY.shape[0], 1))/DY.shape[0]
        self.N = DY.shape[0]
        for id in np.arange(self.T):
            dtree, y_bag_hat, y_bag, oob_score = self.ensemble_trees(
                DX, DY, x_names, w, metric_class, feature_samplesize, max_depth, min_size, inprints)
            self.dectrees.append(dtree)
            self.oobs.append(oob_score)

            e_w = np.sum(w[y_bag_hat != y_bag])/np.sum(w)
            e_w = max(1E-6, min(e_w, 999999E-6))

            if e_w < 0.5:
                alpha = 0.5*np.log((1-e_w)/(e_w))
                self.alphas.append(alpha)

                # Schapire vanilla update rule
                w[y_bag_hat == y_bag] = w[y_bag_hat == y_bag]*np.exp(-alpha)
                w[y_bag_hat != y_bag] = w[y_bag_hat != y_bag]*np.exp(alpha)
                w = w/np.sum(w)

                # MIT update rule
                # w[y_bag_hat == y_bag] =  w[y_bag_hat == y_bag]/(2*(1-e_w))
                # w[y_bag_hat != y_bag] =  w[y_bag_hat != y_bag]/(2*(e_w))

                # OAS update rule
                # w[y_bag_hat == y_bag] =  0.5*w[y_bag_hat == y_bag]
                # w[y_bag_hat != y_bag] =  0.5*(1+e_w)*w[y_bag_hat != y_bag]/(e_w)

                if inprints == 1:
                    print(f"Normalization: {np.sum(w)}")
                    print(f"tree={id+1}\n")
            else:
                self.alphas.append(0)
                if inprints == 1:
                    print(f"tree={id+1}\n")

            # bloss = (y_bag_hat == y_bag)
            # bloss_sign = np.where(bloss == 1, 1, -1)
            # wt_closs = alpha*bloss_sign
            # w = (w)*np.exp(-wt_closs)
            # w = w/np.sum(w)
            # print(np.sum(w))

        self.oobs_mean = np.mean(np.array(self.oobs))
        # out-of-bag estimate
        # print(f"Forest => OOB Mean-Estimate: {self.oobs_mean:.2g}")

    def infer_ensemble(self, data_x, data_y=None):
        pred_y = []
        alphas = np.array(self.alphas).reshape((len(self.alphas), 1))
        for data_x_row in data_x:
            ytemp = []
            for dectrees in self.dectrees:
                decision = dectrees.classify(dectrees.nodes, data_x_row)
                ytemp.append(decision)
            # select weighted average of predictions
            ytempn = np.array(ytemp).reshape((len(ytemp), 1))
            ytempn = np.where(ytempn == 0, -1, 1)
            v = alphas.T@ytempn
            
            # hard: linear
            # yhat_avg = np.sign(v)
            # yhat_avg = np.where(yhat_avg == -1, 0, 1)
            
            # # soft: logistic
            yhat_avg = stdlogistic(v)
            decbnds = 0.5
            yhat_avg = np.where(yhat_avg >= decbnds, 1, 0)
            
            
            # yhat_avg = np.abs(ywt/np.sum(alphas))
            # print(yhat_avg)
            pred_y.append(yhat_avg)

        pred_y = np.array(pred_y).reshape((len(pred_y), 1))

        if not(data_y is None):
            batchlen = len(data_y)
            counts = np.sum(pred_y == data_y)
            # 0/1 accuracy (fractional)
            accs = counts/batchlen
            # 0/1 misses
            misses = batchlen - counts
            print(f"AdaBoost => class accuracy: {accs:.2g}, misses: {misses}")
            return pred_y, accs, misses
        else:
            return pred_y

        
