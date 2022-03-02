from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName('Examine data about passengers on the Titanic') \
    .getOrCreate()

rawdata = spark.read.format('csv').option("header", "true").load('../../datasets/titanic.csv')

# print(rawdata.show(5))

from pyspark.sql.functions import col

dataset = rawdata.select(col('Survived').cast('float'),
                         col('Pclass').cast('float'),
                         col('Sex'),
                         col('Age').cast('float'),
                         col('Fare').cast('float'),
                         col('Embarked'))
# print(dataset.show(5))

dataset = dataset.replace('?', None).dropna(how='any')

from pyspark.ml.feature import StringIndexer

dataset = StringIndexer(
    inputCol='Sex',
    outputCol='Gender',
    handleInvalid='keep').fit(dataset).transform(dataset)

dataset = StringIndexer(
    inputCol='Embarked',
    outputCol='Boarded',
    handleInvalid='keep').fit(dataset).transform(dataset)

# print(dataset.show(5))

dataset = dataset.drop('Sex').drop('Embarked')
# print(dataset.show(5))

requiredFeatures = [
    'Survived',
    'Pclass',
    'Age',
    'Fare',
    'Gender',
    'Boarded'
]

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=requiredFeatures, outputCol='features')

transformed_data = assembler.transform(dataset)

# print(transformed_data.show(5))

from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=8, seed=3)
model = kmeans.fit(transformed_data)

clusteredData = model.transform(transformed_data)

from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator()
# silhouette = measure of how similar an object is to its own cluster
# 1 is ideal value
silhouette = evaluator.evaluate(clusteredData)
print('Silhoette with Squared euclidean distance =', silhouette)

# No fk(cluster specified
# every Cluster center is an array with value for every features specified
# Array rank = k,(num of features)
centers = model.clusterCenters()
print('Cluster Centers: ')
for center in centers:
    print(center)

# Transform adds prediction column with value 0 to 4(cluster number)
clusteredData.show(5)

from pyspark.sql.functions import *

dataset.select(
    avg('Survived'),
    avg("Pclass"),
    avg('Age'),
    avg('Fare'),
    avg('Gender'),
    avg('Boarded')).show(5)

# Observation
# Survival rates is very high for :
# 1. All first class passengers
# 2. Fare paid is very high
# 3. Gender is mostly female
clusteredData.groupby('prediction').agg(
    avg('Survived'),
    avg("Pclass"),
    avg('Age'),
    avg('Fare'),
    avg('Gender'),
    avg('Boarded'),
    count('prediction')).show()

clusteredData.filter(clusteredData.prediction == 1).show(5)
