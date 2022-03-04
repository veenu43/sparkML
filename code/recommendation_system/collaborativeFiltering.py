from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName('Use Collaborative Filtering for movie recommendations') \
    .getOrCreate()

rawdata = spark.read.format('csv').option("header", "true").load('../../datasets/movielens/ratings.csv')

# rawdata.show(5)

from pyspark.sql.functions import col

dataset = rawdata.select(col('userId').cast('int'),
                         col('movieId').cast('int'),
                         col('rating').cast('float'))

dataset.show(5)
# print(dataset.select('rating').toPandas().describe())

(trainingData, testData) = dataset.randomSplit([0.8, 0.2])

from pyspark.ml.recommendation import ALS

# coldSTartStrategy == how to handle new or unknown users or items during prediction
# drop for these cases
als = ALS(
    maxIter=5,
    regParam=0.1,
    userCol='userId',
    itemCol='movieId',
    ratingCol='rating',
    coldStartStrategy='drop'
)

model = als.fit(trainingData)
predictions = model.transform(testData)
# predictions.show(5)

# compare the distribution of actual and predicted ratings in test data
print(predictions.select('rating', 'prediction').toPandas().describe())

from pyspark.ml.evaluation import RegressionEvaluator

# Evaluate an ALS model using explicitly stated with its RMSE
evaluator = RegressionEvaluator(
    metricName='rmse',
    labelCol='rating',
    predictionCol='prediction'
)

rmse = evaluator.evaluate(predictions)
print("RMSE = ", rmse)

# top 3 movie recommendation for users
# recommendations are a list of (movieID,rating) tuples
userRecsAll = model.recommendForAllUsers(3)
print(userRecsAll)
print(userRecsAll.toPandas().head())

# Also view the users most likely to like a movie.This function returns the top 3 users for each movie
movieRecsAll = model.recommendForAllItems(3)
print(movieRecsAll)
print(movieRecsAll.toPandas().head())

from pyspark.sql.types import IntegerType

# to get recommendations for a specific set of users,we need to create a dataframe containing only those users
usersList = [148, 463, 267]
usersDF = spark.createDataFrame(usersList, IntegerType()).toDF('userId')
usersDF.take(3)

# top 5 movie recommendations are a list of (movieID,rating) tuples for user set
userRecs = model.recommendForUserSubset(usersDF, 5)
print(userRecs)
print(userRecs.toPandas().head())

# Extract recommendations for single user into a Spark dataframe
userMovieList = userRecs.filter(userRecs.userId == 148).select('recommendations')

# One row for each user containing recommendations for that user
# each row in turn contains a recommendations column which is a list of Rows
userMovieList.collect()
moviesList = userMovieList.collect()[0].recommendations
print(moviesList)

# create a DF containing the movieID and rating in separate columns
moviesDF = spark.createDataFrame(moviesList)
moviesDF.toPandas()


