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
import time
%matplotlib inline

# read in small dataset (rememeber to adjust the path)
#data_small = pd.read_csv('/Users/Xiu/Documents/dsga1004/final-project-group_80/ml-latest-small/ratings.csv')[['userId', 'movieId', 'rating']]
train_small   = pd.read_csv('/Users/Xiu/Documents/dsga1004/final-project-group_80/train_small.csv')[['userId', 'movieId', 'rating']]
test_small  = pd.read_csv('/Users/Xiu/Documents/dsga1004/final-project-group_80/test_small.csv')[['userId', 'movieId', 'rating']]

# rename cols
train_small.rename(columns = {'userId':'user', 'movieId':'item'}, inplace = True)
test_small.rename(columns = {'userId':'user', 'movieId':'item'}, inplace = True)

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
#Start timer here
tic = time.perf_counter()
all_recs.append(eval('ALS', algo_als, train_small, test_small))
#End timer here 
toc = time.perf_counter() 
print(f"Finished program in {toc - tic:0.4f} seconds")
all_recs = pd.concat(all_recs, ignore_index=True)
all_recs.head()

all_recs.shape
test_small = pd.concat(test_small, ignore_index=True)

rla = topn.RecListAnalysis()
rla.add_metric(topn.precision)
results = rla.compute(all_recs, test_small)
results.head()









