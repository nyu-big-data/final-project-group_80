# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:52:55 2022

@author: howel
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import os
from IPython.display import display
import pandas as pd 
import numpy as np
import itertools
import sklearn.model_selection
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import collect_list
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
     
    
    # Load the data into DataFrame

    #small = spark.read.csv(f'hdfs:/user/{netID}/train_big.csv', mode="DROPMALFORMED", inferSchema=True, header = True)
    #Create Table View
    #small.createOrReplaceTempView('small')

    # large = spark.read.csv(f'hdfs:/user/{netID}/ratings_large.csv', mode="DROPMALFORMED", inferSchema=True, header = True)
    # #Create Table View
    # large.createOrReplaceTempView('large')

    trainsmall = spark.read.csv(f'hdfs:/user/{netID}/train_small.csv', mode="DROPMALFORMED", inferSchema=True, header = True)
    trainsmall.createOrReplaceTempView('trainsmall')
    trainsmall.printSchema()
    
    valsmall = spark.read.csv(f'hdfs:/user/{netID}/val_small.csv', mode="DROPMALFORMED", inferSchema=True, header = True)
    valsmall.printSchema()
    
    # trainsmall.show()
    
    
    testsmall = spark.read.csv(f'hdfs:/user/{netID}/test_small.csv', mode="DROPMALFORMED", inferSchema=True, header = True)
    testsmall.printSchema()
    
    ##########################Popularity Baseline Model###################################

    popularity_small = spark.sql('SELECT movieId, AVG(rating) AS average_rating, count(userId) AS num_review FROM trainsmall GROUP BY movieId HAVING num_review > 30 ORDER BY average_rating DESC LIMIT 100 ')
    print('Popularity model, small')
    popularity_small.show()

    #print(list(popularity_small.select('movieId').toPandas()['movieId']))
    TopMovieList = list(popularity_small.select('movieId').toPandas()['movieId'])
    
    watch = testsmall.groupby('userId').agg(collect_list('movieId')).alias('watchedmovies').sort('userId').collect()
    testsmallprocessed = spark.createDataFrame(watch).select('collect_list(movieId)')
    watched = list(testsmallprocessed.select('collect_list(movieId)').toPandas()['collect_list(movieId)'])
    
    TopMovieList = [TopMovieList]*len(watched)

    
    prediction = spark.sparkContext.parallelize(list(zip(watched,TopMovieList)))
    
    
    metrics = RankingMetrics(prediction)
    
    print('Metrics')
    print("Precision at 1 ", metrics.precisionAt(1))
    print("Precision at 5 " ,metrics.precisionAt(5))
    print("Precision at 10 ", metrics.precisionAt(10))
    print("Precision at 100 ", metrics.precisionAt(100))
    print("Mean Average Precision: ")
    print(metrics.meanAveragePrecision)
    
    print("ndcgAt at 1: ", metrics.ndcgAt(1))
    print("ndcgAt at 5: " ,metrics.ndcgAt(5))
    print("ndcgAt at 10: ", metrics.ndcgAt(10))
    print("ndcgAt at 100: ", metrics.ndcgAt(100))
    print("ndcgAt: ")
    print(metrics.ndcgAt)

    
    # popularity_large = spark.sql('SELECT movieId, AVG(rating) AS average_rating, count(userId) AS num_review FROM large GROUP BY movieId HAVING num_review > 30 ORDER BY average_rating DESC LIMIT 100 ')    
    # print('Popularity model, large') 
    # popularity_large.show()

    ######################Latent Factor Model##########################################
    
    def Extract(lst):
        return [item[0] for item in lst]  
    
    ranklist = [5, 10, 15, 20]
    iterlist = [3, 5, 10, 15]
    regparamlist = [.05, .1, 1]
    
    storemap = []
    storecount = []
    
    #Changes this to validation set. If we comment this whole latent factor set out, then we are using the testset.
    
    watch = valsmall.groupby('userId').agg(collect_list('movieId')).alias('watchedmovies').sort('userId').collect()
    testsmallprocessed = spark.createDataFrame(watch).select('collect_list(movieId)')
    watched = list(testsmallprocessed.select('collect_list(movieId)').toPandas()['collect_list(movieId)'])
    
    for x in ranklist:
        for y in iterlist:
            for z in regparamlist:
                print(x)
                print(y)
                print(z)
                als = ALS(rank = x, maxIter=y, regParam=z,userCol="userId", itemCol="movieId", ratingCol="rating",nonnegative = True, implicitPrefs = False, coldStartStrategy="drop")
                model = als.fit(trainsmall)
                users = valsmall.select(als.getUserCol()).distinct()
                userSubsetRecs = model.recommendForUserSubset(users, 100)
                userSubsetRecs = userSubsetRecs.sort('userId')
                usersubsettest=list(userSubsetRecs.toPandas()['recommendations'])
                storage = []
                for i, j in enumerate(usersubsettest): 
                    storage.append(Extract(usersubsettest[i]))
                prediction = spark.sparkContext.parallelize(list(zip(watched,storage)))
                metrics = RankingMetrics(prediction)
                print("Mean Average Precision: ", metrics.meanAveragePrecision)
                print("ndcgAt at 100: ", metrics.ndcgAt(100))
                storemap.append(metrics.meanAveragePrecision)
                storecount.append("Rank: " + str(x)+ "Iterations: " + str(y)+ "Regularization Parameter: " +str(z))
    
    indextouse = np.argmin(storemap)
    print(np.amin(storemap))
    print(storecount[indextouse])
    
    #######################################################################################################
    
    
    # als = ALS(maxIter=3, regParam=0.1,rank =5, userCol="userId", itemCol="movieId", ratingCol="rating",nonnegative = True, implicitPrefs = False, coldStartStrategy="drop")

    # model = als.fit(trainsmall)
    
    # users = testsmall.select(als.getUserCol()).distinct()
    # userSubsetRecs = model.recommendForUserSubset(users, 100)
    
    
    # userSubsetRecs = userSubsetRecs.sort('userId')
    
    # userSubsetRecs.limit(100).show()
    
    # usersubsettest=list(userSubsetRecs.toPandas()['recommendations'])
    
    # #count = len(usersubsettest)

  
    
    # storage = []

    # for i, j in enumerate(usersubsettest): 
    #     storage.append(Extract(usersubsettest[i]))

    # #print(storage)
    
    # prediction = spark.sparkContext.parallelize(list(zip(watched,storage)))
    
    # metrics = RankingMetrics(prediction)
    
    # print('Metrics')
    # print("Precision at 1: ", metrics.precisionAt(1))
    # print("Precision at 5: " ,metrics.precisionAt(5))
    # print("Precision at 10: ", metrics.precisionAt(10))
    # print("Precision at 100: ", metrics.precisionAt(100))
    # print("Mean Average Precision: ")
    # print(metrics.meanAveragePrecision)
    
    # print("ndcgAt at 1: ", metrics.ndcgAt(1))
    # print("ndcgAt at 5: " ,metrics.ndcgAt(5))
    # print("ndcgAt at 10: ", metrics.ndcgAt(10))
    # print("ndcgAt at 100: ", metrics.ndcgAt(100))
    # print("ndcgAt: ")
    # print(metrics.ndcgAt)
    
    
    print('Values for #4:')
    # userIDsubsettest=list(userSubsetRecs.toPandas()['userId'])
    
    # BadUserRankings = []
    # LowestValues = []
    # for a in range(len(watched)):
    #     prediction = spark.sparkContext.parallelize(list(zip(watched[a],storage[a])))
    #     metrics = RankingMetrics(prediction)
    #     LowestValues.append(metrics.meanAveragePrecision)
    #     BadUserRankings.append(userIDsubsettest[a])
    
    # pandasdf = pd.DataFrame(list(zip(BadUserRankings, LowestValues)),columns=['USERSLIST','MAPVAL'])
    
    # pandasdf = pandasdf.sort_values(by='MAPVAL', ascending=False) 
    
    # display(pandasdf)
    #pandasdf.show()
    
    
    #print(usersubsettest.iloc[1])
    
    
    
    #usersubsettest.to_csv('/home/hl4631/final-project-group_80',index=True, encoding='utf-8')

    
    
    
    # predictions = model.transform(testsmall)
    
    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
    #                                 predictionCol="prediction")
    # rmse = evaluator.evaluate(predictions)
    # print("Root-mean-square error = " + str(rmse))
    
    # # Generate top 10 movie recommendations for each user
    # userRecs = model.recommendForAllUsers(10)
    # # Generate top 10 user recommendations for each movie
    # movieRecs = model.recommendForAllItems(10)
    
    # # Generate top 10 movie recommendations for a specified set of users
    # users = testsmall.select(als.getUserCol()).distinct().limit(3)
    # userSubsetRecs = model.recommendForUserSubset(users, 10)
    # # Generate top 10 user recommendations for a specified set of movies
    # movies = testsmall.select(als.getItemCol()).distinct().limit(3)
    # movieSubSetRecs = model.recommendForItemSubset(movies, 10)
    
    # RDDUse = predictions.RDD()
    
    # # $example off$
    # userRecs.show()
    # movieRecs.show()
    # userSubsetRecs.show()
    # movieSubSetRecs.show()
    #######################################################################

    spark.stop()
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
