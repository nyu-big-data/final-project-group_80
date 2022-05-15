import os
from IPython.display import display
import pandas as pd 
import numpy as np
import itertools
import sklearn.model_selection

# read in ratings data
ratings = pd.read_csv(('ml-latest-small/ratings.csv'))
#ratings = pd.read_csv(('ml-latest/ratings.csv'))

np.random.seed(1)
# get the unique users
users = np.unique(ratings['userId'])

#We will create a function that takes the index of 60% of viewing data. 
def save_as_train(user):
    fullset = ratings.loc[ratings['userId'] == user]
    subset = fullset.sample(frac = 0.6, random_state=1)
    return subset.index

#We will apply the function to each user, taking the index of 60% of data from each user as training.
training_index = list(map(save_as_train, users))
training_index = list(itertools.chain(*training_index))

#We will save the ratings corresponding those index as train.
train_set = ratings.loc[training_index]
#We will now split the rest of the ratings into test and validation.
val_test = ratings.drop(training_index)

#We split the users into two group. 
split_users = np.random.choice(users_small,np.ceil(len(users)*0.5).astype(int),replace = False)
#For one, the remaining 40% of data will be test.
val_set = val_test.loc[~val_test['userId'].isin(split_users)]
#For the other, the remaining 40% of data will be validation.
test_set = val_test.loc[val_test['userId'].isin(split_users)]

# save to home 
train_set.to_csv('ratings_data/train_small.csv', index = False)  
val_set.to_csv('ratings_data/val_small.csv', index = False)  
test_set.to_csv('ratings_data/test_small.csv', index = False)  

#train_set.to_csv('ratings_data/train_large.csv')  
#val_set.to_csv('ratings_data/val_large.csv')  
#test_set.to_csv('ratings_data/test_large.csv')  