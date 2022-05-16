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


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
     
    
    # Load the data into DataFrame

    small = spark.read.csv(f'hdfs:/user/{netID}/train_small.csv', mode="DROPMALFORMED", inferSchema=True, header = True)
    #Create Table View
    small.createOrReplaceTempView('small')

    # large = spark.read.csv(f'hdfs:/user/{netID}/ratings_large.csv', mode="DROPMALFORMED", inferSchema=True, header = True)
    # #Create Table View
    # large.createOrReplaceTempView('large')

    trainsmall = spark.read.csv(f'hdfs:/user/{netID}/train_small.csv', mode="DROPMALFORMED", inferSchema=True, header = True)
    trainsmall.printSchema()
    
    # trainsmall.show()
    
    
    testsmall = spark.read.csv(f'hdfs:/user/{netID}/test_small.csv', mode="DROPMALFORMED", inferSchema=True, header = True)
    testsmall.printSchema()
    
    ##########################Popularity Baseline Model###################################

    popularity_small = spark.sql('SELECT movieId, AVG(rating) AS average_rating, count(userId) AS num_review FROM small GROUP BY movieId HAVING num_review > 30 ORDER BY average_rating DESC LIMIT 100 ')
    print('Popularity model, small')
    #popularity_small.show()

    print(list(popularity_small.select('movieId').toPandas()['movieId']))
    TopMovieList = list(popularity_small.select('movieId').toPandas()['movieId'])
    
    watch = trainsmall.groupby('userId').agg(collect_list('movieId')).alias('watchedmovies').sort('userId').collect()
    trainsmallprocessed = spark.createDataFrame(watch).select('collect_list(movieId)')
    watched = list(trainsmallprocessed.select('collect_list(movieId)').toPandas()['collect_list(movieId)'])
    
    TopMovieList = [TopMovieList]*len(watched)

    
    prediction = spark.sparkContext.parallelize(list(zip(watched,TopMovieList)))
    
    metrics = RankingMetrics(prediction)
    
    print('Metrics')
    print(metrics.precisionAt(1))
    print(metrics.precisionAt(5))
    print(metrics.precisionAt(10))
    print(metrics.meanAveragePrecision)
    
    # popularity_large = spark.sql('SELECT movieId, AVG(rating) AS average_rating, count(userId) AS num_review FROM large GROUP BY movieId HAVING num_review > 30 ORDER BY average_rating DESC LIMIT 100 ')    
    # print('Popularity model, large') 
    # popularity_large.show()

    #######################Latent Factor Model##########################################
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",coldStartStrategy="drop")
    model = als.fit(trainsmall)
    
    users = trainsmall.select(als.getUserCol()).distinct()
    userSubsetRecs = model.recommendForUserSubset(users, 100)
    
    
    userSubsetRecs.sort('userId').show()
    
    usersubsettest=userSubsetRecs.limit(5).toPandas()

    print(os.getcwd())
    print(usersubsettest[1])
    usersubsettest.to_csv('/home/hl4631/final-project-group_80',index=True, encoding='utf-8')

    
    
    
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
