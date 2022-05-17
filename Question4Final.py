# -*- coding: utf-8 -*-
"""
Created on Tue May 17 01:09:31 2022

@author: howel
"""

cd Downloads

from lenskit.datasets import ML100K
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit import topn
import time
import pandas as pd

data_small = pd.read_csv('fullratings.csv')[['userId', 'movieId', 'rating']]


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
#Start timer here
tic = time.perf_counter()
for train, test in xf.partition_users(data_small, 5, xf.SampleFrac(0.2)):
    test_small.append(test)
    all_recs.append(eval('ALS', algo_als, train, test))
#End timer here
toc = time.perf_counter()
print(f"Finished program in {toc - tic:0.4f} seconds")






all_recs = pd.concat(all_recs, ignore_index=True)
all_recs.head()



test_small = pd.concat(test_small, ignore_index=True)


rla = topn.RecListAnalysis()
rla.add_metric(topn.precision)
results = rla.compute(all_recs, test_small)

results.to_csv('resultsprecisest.csv')



Mean = test_small.iloc[:,2].mean()

user1=data_small.query('user==56341')
user1.iloc[:,2].mean()

user2=data_small.query('user==134596')
user2.iloc[:,2].mean()


user3=data_small.query('user==63783')
user3.iloc[:,2].mean()


user4=data_small.query('user==123100')
user4.iloc[:,2].mean()


user5=data_small.query('user==117490')
user5.iloc[:,2].mean()


user6=data_small.query('user==42494')
user6.iloc[:,2].mean()

Mean = test_small.iloc[:,2]

ListofRankings = [] 
ListofMeans = []

for i,j in results.index:
    ListofRankings.append(j)
    
    
for i in ListofRankings:
    userval = f"user=={i}"
    ListofMeans.append(data_small.query(userval).iloc[:,2].mean())
    

listofaverages = pd.read_csv('CryptoBoyyy.csv')

queries = "select avg(rating) from ratings group by userId;"

listofaverages.quantile(0.1)
listofaverages.quantile(0.2)
listofaverages.quantile(0.8)
listofaverages.quantile(0.9)

morequeries = "select tag,count(tag) from tags where movieId in (select movieId from ratings where userId = 56341) group by tag order by count(tag) DESC"

