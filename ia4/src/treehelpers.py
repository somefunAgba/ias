# somefuno@oregonstate.edu
# Helper Functions API

import numpy as np
import pandas as pd
np.set_printoptions(precision=4)

# random number generator
rng = np.random.default_rng(2021)

# set operation  
def differ(a, b):
    # returns the list of elements in the left iterable object
    # not in the right iterable object
    return list(set(a) - set(b))

# class cross entropy
def ent_class(y, w=None, N=None):
    uclass = np.unique(y)
    ent = 0
    for uy in uclass:
        # counts
        if w is None:
            n_uy = np.sum(y == uy)
            p = n_uy/len(y)
        else:
            p = np.sum(w[y == uy])
            # p = p*(N/len(y)) # seems log takes care of this, so no need.
            p = np.where(p > 1.0e-10, p, 1.0e-10)
        ent -= p*np.log2(p)

    return ent

# class gini
def gini_class(y, w=None, N=None):
    uclass = np.unique(y)
    gini = 1
    for uy in uclass:
        # counts
        if w is None:
            n_uy = np.sum(y == uy)
            p = n_uy/len(y)
        else:
            p = np.sum(w[y == uy])
            p = p*(N/len(y))
        
        gini -= (p)**2

    return gini


# sampling
def sample_population(pop_size, samp_size, withreplace=False, p=None):
    '''
    pop_size: population size
    samp_size: samp_size
    withreplace: True or False
    '''
    # population_size = 10
    # sample_size = 4
    # population size must be > sample_size
    # choose sample out of population
    if samp_size > pop_size:
        raise ValueError('expects: left input >= right input')
    
    if p is None:
        sel_choice = rng.choice(pop_size, size=samp_size, replace=withreplace)
    else:
        sel_choice = rng.choice(pop_size, size=samp_size, replace=withreplace,p=p)

    return sel_choice
  
  
  
# bagging: row sampling
def bagging(x, y, withreplace=True, p=None, skipbag=0):
    '''
    returns x_bag, y_bag, x_oob, y_oob
    '''

    pop_size = x.shape[0]
    samp_size = pop_size

    '''
    pop_size: population size
    samp_size: samp_size
    withreplace: True or False
    '''
    # population_size = 10
    # sample_size = 4
    # population size must be > sample_size
    # choose sample out of population
    if samp_size > pop_size:
        raise ValueError('expects: left input >= right input')

    # all ids
    full_ids = np.arange(pop_size)

    if skipbag == 0:
      # in-bag ids
      if not(p is None):
        p = p.flatten()
        sel_ids = rng.choice(
          pop_size, samp_size, withreplace, p)
      else:
        sel_ids = rng.choice(
          pop_size, samp_size, withreplace)
    
      # out of bag ids
      oob_ids = np.array(differ(full_ids, sel_ids))
    else: 
      sel_ids = full_ids
      oob_ids = full_ids

    x_bag = x[sel_ids, :]
    y_bag = y[sel_ids, :]
    x_oob = x[oob_ids, :]
    y_oob = y[oob_ids, :]

    return x_bag, y_bag, x_oob, y_oob


# probability distribution for feature resampling/bagging using info gain
def feat_split_dist(DX, DY, x_names, split_metric=ent_class, w=None, m=0, N=None):

    # class counts for given data - root
    y = DY
    if split_metric == ent_class:
        ent_yroot = split_metric(y, w, N)

    # given rows of data
    # root_rows = {'DX': DX, 'DY': y}
    # y = root_rows['DY']
    infogains = []  # track best infogain in each x feature
    uvalx_gain = []  # track unique val w.r.t to best infogain in x feature

    for x in DX.T:

        # unique values in each column (feature) of the data
        unique_x = np.unique(x)

        # in: data, take each row of feature x,  its unique features ux
        # binary split of rows based on question on a feature
        # for rowx in x:

        # conditional entropy of y given this x column feature
        enty_c_x = 0
        # info gain on y given this x column feature
        if split_metric == ent_class:
            infogain_x = 0
        elif split_metric == gini_class:
            infogain_x = 1
        ux_best = 0
        for ux in unique_x:
            # question: this unique value -> this feature
            qat = (x == ux)  # true parts # >=, <=
            qaf = ~(qat)    # false parts

            # skip this split quest
            # if either true parts or false parts is empty
            if len(qat)*len(qaf) == 0:
                ux_best = ux
                if split_metric == ent_class:
                    infogain_x = 0
                elif split_metric == gini_class:
                    infogain_x = 1
                    
                continue

            # print(f"Is {x} == {ux}? {qat}")

            # full data split based on this unique value -> feature
            # leaf_trows = {'DX' : DX[qat], 'DY' : y[qat]}
            # leaf_frows = {'DX' : DX[qaf], 'DY' : y[qaf]}

            # counts: this unique value -> feature
            n_t = np.sum(qat)  # print(ux,n_t \equiv n_ux)
            # likelihood of the truth of unique value
            pxt = n_t/len(x)
            # corresponding class given the truth of this unique value
            y_uxt = y[qat]
            if (w is None):
                w_uxt = None
            else:
                w_uxt = w[qat]
                pxt = np.sum(w_uxt)
                
            # condit. entropy of y given truth of this unique value -> feature
            ent_y_c_uxt = split_metric(y_uxt, w_uxt, N)

            # - altA start
            # enty_c_x +=  pxt*ent_y_c_uxt
            # - altA end

            # - altB start
            # corresponding class given the falsity of this unique value
            y_uxf = y[qaf]
            pxf = (1-pxt)
            if (w is None):
                w_uxf = None
            else:
                w_uxf = w[qaf]
                pxf = np.sum(w_uxf)
            # condit. entropy of y given falsity of this unique value -> feature
            ent_y_c_uxf = split_metric(y_uxf, w_uxf, N)
            # condit. entropy of y given this unique value -> feature
            enty_c_ux = pxt*ent_y_c_uxt + pxf*ent_y_c_uxf

            # info gain on y given this this unique value in this x column/feature
            if split_metric == ent_class:
                infogain_ux = ent_yroot - enty_c_ux
                if infogain_ux > infogain_x:
                    infogain_x = infogain_ux
                    ux_best = ux
            elif split_metric == gini_class:
                infogain_ux = enty_c_ux
                if infogain_ux <= infogain_x:
                    infogain_x = infogain_ux
                    ux_best = ux
            # - altB end

        # - altB start
        infogains.append(infogain_x)
        uvalx_gain.append(ux_best)
        # - altB end

        # - altA start
        # information gain for split on this column/feature
        # infogain_x = ent_yroot - enty_c_x
        # infogains.append(infogain_x)
        # - altA end
    if split_metric == ent_class:
        pfeat_dist = infogains/np.sum(infogains)
    else:
        gains = 1 - np.array(infogains)
        pfeat_dist = gains/np.sum(gains)
        pfeat_dist.tolist()
    return pfeat_dist

# greedily search for best decision split on data 
# by:
# iterating over every unique value in each feature in given data
# - uses information gain to decide
#
# Outputs: Best feature for split, with its id, name, val and ig
# Information Gain = ig
# Unique Value in Feature = val
# Column ID in data = id
# Column Name in data = name
def search_bestsplit(DX, DY, x_names, split_metric=ent_class, w=None, m=0, N=None, colw=None):

    # class counts for given data - root
    y = DY
    if split_metric == ent_class:
        ent_yroot = split_metric(y, w, N)

    # given rows of data
    # root_rows = {'DX': DX, 'DY': y}
    # y = root_rows['DY']
    infogains = []  # track best infogain in each x feature
    uvalx_gain = []  # track unique val w.r.t to best infogain in x feature

    if m > 0:
        # number of random features to subsample out of feature population
        m_pop = DX.shape[1]
        m_samp = m
        if m_samp > m_pop:
            print("FYI: sample_size exceeded population | fixed.")
            m_samp = m_pop
        if colw is None:
            selfeats = np.sort(sample_population(m_pop, m_samp, withreplace=True))
        else:
            # repeat selection as much as possible to randomly select relevant features with high prob (w.h.p)
            fbag = []; psums = 0
            for i in range(0,16):
                selfeats = np.sort(sample_population(m_pop, m_samp, withreplace=True, p=colw))
                selmetric = np.sum(colw[selfeats])
                if selmetric > psums:
                    fbag = selfeats
                    psums = selmetric 
            selfeats = fbag
                
        DX_sel = DX[:, selfeats]
        x_names_sel = []
        for idn in selfeats:
            x_names_sel.append(x_names[idn])
        x_names = x_names_sel.copy()
    else:
        DX_sel = DX

    for x in DX_sel.T:

        # unique values in each column (feature) of the data
        unique_x = np.unique(x)

        # in: data, take each row of feature x,  its unique features ux
        # binary split of rows based on question on a feature
        # for rowx in x:

        # conditional entropy of y given this x column feature
        enty_c_x = 0
        # info gain on y given this x column feature
        if split_metric == ent_class:
            infogain_x = 0
        elif split_metric == gini_class:
            infogain_x = 1
        ux_best = 0
        for ux in unique_x:
            # question: this unique value -> this feature
            qat = (x == ux)  # true parts # >=, <=
            qaf = ~(qat)    # false parts

            # skip this split quest
            # if either true parts or false parts is empty
            if len(qat)*len(qaf) == 0:
                ux_best = ux
                if split_metric == ent_class:
                    infogain_x = 0
                elif split_metric == gini_class:
                    infogain_x = 1
                    
                continue

            # print(f"Is {x} == {ux}? {qat}")

            # full data split based on this unique value -> feature
            # leaf_trows = {'DX' : DX[qat], 'DY' : y[qat]}
            # leaf_frows = {'DX' : DX[qaf], 'DY' : y[qaf]}

            # counts: this unique value -> feature
            n_t = np.sum(qat)  # print(ux,n_t \equiv n_ux)
            # likelihood of the truth of unique value
            pxt = n_t/len(x)
            # corresponding class given the truth of this unique value
            y_uxt = y[qat]
            if (w is None):
                w_uxt = None
            else:
                w_uxt = w[qat]
                pxt = np.sum(w_uxt)
                
            # condit. entropy of y given truth of this unique value -> feature
            ent_y_c_uxt = split_metric(y_uxt, w_uxt, N)

            # - altA start
            # enty_c_x +=  pxt*ent_y_c_uxt
            # - altA end

            # - altB start
            # corresponding class given the falsity of this unique value
            y_uxf = y[qaf]
            pxf = (1-pxt)
            if (w is None):
                w_uxf = None
            else:
                w_uxf = w[qaf]
                pxf = np.sum(w_uxf)
            # condit. entropy of y given falsity of this unique value -> feature
            ent_y_c_uxf = split_metric(y_uxf, w_uxf, N)
            # condit. entropy of y given this unique value -> feature
            enty_c_ux = pxt*ent_y_c_uxt + pxf*ent_y_c_uxf

            # info gain on y given this this unique value in this x column/feature
            if split_metric == ent_class:
                infogain_ux = ent_yroot - enty_c_ux
                if infogain_ux > infogain_x:
                    infogain_x = infogain_ux
                    ux_best = ux
            elif split_metric == gini_class:
                infogain_ux = enty_c_ux
                if infogain_ux <= infogain_x:
                    infogain_x = infogain_ux
                    ux_best = ux
            # - altB end

        # - altB start
        infogains.append(infogain_x)
        uvalx_gain.append(ux_best)
        # - altB end

        # - altA start
        # information gain for split on this column/feature
        # infogain_x = ent_yroot - enty_c_x
        # infogains.append(infogain_x)
        # - altA end

    # print(f"DX: {infogains}")
    if split_metric == ent_class:
        infogain_best = np.amax(infogains)
        infogain_best_x_id = np.argmax(infogains)
    elif split_metric == gini_class:
        infogain_best = np.amin(infogains)
        infogain_best_x_id = np.argmin(infogains)

    # - altB start
    x_name_best = x_names[infogain_best_x_id]
    uvalx_best = uvalx_gain[infogain_best_x_id]
    # - altB end

    # print("Information Gain: I.G")
    # - altA start
    # print(f"I.G (best split) : {infogain_best}, id: {infogain_best_x_id}, name: {x_names[infogain_best_x_id]}")
    # - altA end

    # - altB start
    # print(
    #     f"I.G (best split) : {infogain_best}, id: {infogain_best_x_id}, name: {x_name_best}, val: {uvalx_best}")
    # - altB end

    return {'name': x_name_best, 'id': infogain_best_x_id, 'val': uvalx_best, 'ig': infogain_best}


# Majority Voting 
def max_class(y, w=None):
    n_uy = []
    uclass = np.unique(y)
    for uy in uclass:
        # counts
        if w is None:
            n_uy.append(np.sum(y == uy))
        else:
            n_uy.append(np.sum(w[y == uy]))

    # decision theory:

    if w is None:
        # return class/output that occurs most
        max_y = uclass[np.argmax(n_uy)]
    else:
        # return weighted output
        uclass = np.where(uclass == 0, -1, 1)
        n_uy = np.array(n_uy)
        max_y = np.sign( uclass.dot(n_uy) )
        max_y = np.where(max_y == -1, 0, 1)

    return max_y