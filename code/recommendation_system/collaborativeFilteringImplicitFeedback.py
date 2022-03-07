from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName('Use Implicit Collaborative Filtering for band recommendations') \
    .getOrCreate()

rawdata = spark.read.format('csv').option("delimiter", "\t").option("header", "true").load(
    '../../datasets/lastfm/user_artists.dat')

# rawdata.show(5)

from pyspark.sql.functions import col

dataset = rawdata.select(col('userID').cast('int'),
                         col('artistID').cast('int'),
                         col('weight').cast('int'))

# dataset.show(5)
# print(dataset.select('weight').toPandas().describe())

from pyspark.sql.functions import stddev, mean, col

# Standardize the weight field using the formula: z = x- u/dev
# z == scale value, u = mean, dev = standard deviation
# Standardization mitigate extereme values
df = dataset.select(mean('weight').alias('mean_weight'), stddev('weight').alias('stdev_weight')) \
    .crossJoin(dataset).withColumn("weight_scaled", (col("weight") - col('mean_weight')) / col('stdev_weight'))
# df.show(5)
# df.toPandas().head()

(trainingData, testData) = df.randomSplit([0.8, 0.2])

from pyspark.ml.recommendation import ALS

# coldSTartStrategy == how to handle new or unknown users or items during prediction
# drop for these cases
als = ALS(
    maxIter=10,
    regParam=0.1,
    userCol='userID',
    itemCol='artistID',
    implicitPrefs=True,
    ratingCol='weight_scaled',
    coldStartStrategy='drop'
)

model = als.fit(trainingData)
predictions = model.transform(testData)
# predictions.show(5)

predictionsPandas = predictions.select('weight_scaled', 'prediction').toPandas()
# print(predictionsPandas.describe())

artistData = spark.read.format('csv').option("delimiter", "\t").option("header", "true").load(
    '../../datasets/lastfm/artists.dat')
# artistData.show(5)

from pyspark.sql.types import IntegerType


def getRecommendationsForUser(userId, numRecs):
    usersDF = spark.createDataFrame([userId], IntegerType()).toDF('userId')
    userRecs = model.recommendForUserSubset(usersDF, numRecs)

    artistsList = userRecs.collect()[0].recommendations
    artistsDF = spark.createDataFrame(artistsList)
    recommendedArtists = artistData.join(artistsDF, artistData.id == artistsDF.artistID) \
        .orderBy('rating', ascending=False) \
        .select('name', 'url', 'rating')
    return recommendedArtists


# Top 10 artists recommendattion for user
print(getRecommendationsForUser(939,10).show(5))

# Look at the bands this user has been listening to by examining our datasets to validate prediction
userArtistRaw = dataset.filter(dataset.userID == 939)
userArtistRaw.show(5)
userArtistInfo = artistData.join(userArtistRaw, artistData.id == userArtistRaw.artistID) \
    .orderBy('weight', ascending=False) \
    .select('name', 'weight')
userArtistInfo.show(5)
