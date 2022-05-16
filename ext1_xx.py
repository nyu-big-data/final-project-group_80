#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:18:18 2022

@author: Xiu
"""

#pip install lenskit
from lenskit.datasets import ML100K
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit import topn
import pandas as pd
%matplotlib inline

# read in small dataset
data_small = pd.read_csv('/Users/Xiu/Documents/dsga1004/final-project-group_80/ml-latest-small/ratings.csv')[['userId', 'movieId', 'rating']]
#val_small   = pd.read_csv('/Users/Xiu/Documents/dsga1004/final-project-group_80/val_small.csv')
#test_small  = pd.read_csv('/Users/Xiu/Documents/dsga1004/final-project-group_80/test_small.csv')[['userId', 'movieId', 'rating']]

# rename cols
data_small.rename(columns = {'userId':'user', 'movieId':'item'}, inplace = True)
#test_small.rename(columns = {'userId':'user', 'movieId':'item'}, inplace = True)

algo_als = als.BiasedMF(10, iterations=3, reg=0.1) #tune later according to the other model

def eval(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    
    # now we run the recommender
    recs = batch.recommend(fittable, users, 100)
    
    # add the algorithm name for analyzability
    recs['Algorithm'] = aname
    return recs

all_recs = []
test_small = []
for train, test in xf.partition_users(data_small, 5, xf.SampleFrac(0.2)):
    test_small.append(test)
    all_recs.append(eval('ALS', algo_als, train, test))

all_recs = pd.concat(all_recs, ignore_index=True)
all_recs.head()

all_recs.shape

test_small = pd.concat(test_small, ignore_index=True)

rla = topn.RecListAnalysis()
rla.add_metric(map)
results = rla.compute(all_recs, test_small)
results.head()










