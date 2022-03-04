from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName('Effects of Dimensionality Reduction when making predictions') \
    .getOrCreate()

rawdata = spark.read.format('csv').option("header", "true").load('../../datasets/day.csv')

# rawdata.show(5)

from pyspark.sql.functions import col

dataset = rawdata.select(col('season').cast('float'),
                         col('yr').cast('float'),
                         col('mnth').cast('float'),
                         col('holiday').cast('float'),
                         col('weekday').cast('float'),

                         col('workingday').cast('float'),
                         col('weathersit').cast('float'),
                         col('temp').cast('float'),
                         col('atemp').cast('float'),
                         col('hum').cast('float'),
                         col('windspeed').cast('float'),
                         col('cnt').cast('float'))
dataset.show(5)

import matplotlib.pyplot as plt
import seaborn as sns

'''
corrmat = dataset.toPandas().corr()
plt.figure(figsize=(7, 7))
sns.set(font_scale=1.0)
sns.heatmap(corrmat,vmax=.8,square=True,annot=True,fmt='.2f',cmap='winter')
plt.show()
'''
featureCols = dataset.columns.copy()
featureCols.remove('cnt')
print(featureCols)

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=featureCols, outputCol='features')
vectorDF = assembler.transform(dataset)
vectorDF.show(5)

(trainingData, testData) = vectorDF.randomSplit([0.8, 0.2])

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(maxIter=100, regParam=1.0, elasticNetParam=0.8, labelCol='cnt', featuresCol='features')
model = lr.fit(trainingData)
print('Training R^2 score = ', model.summary.r2)
print('Training RMSE ', model.summary.rootMeanSquaredError)

predictions = model.transform(testData)
# predictions.show(5)

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol='cnt', predictionCol='prediction', metricName='r2')
rsquare = evaluator.evaluate(predictions)
print("Test R^2 score =%g" % rsquare)

evaluator = RegressionEvaluator(labelCol='cnt', predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(predictions)
print("Test RMSE = ", rmse)

predictionsPandas = predictions.toPandas()
plt.plot(predictionsPandas['cnt'], label=['Actual'])
plt.plot(predictionsPandas['prediction'], label=['predicted'])
plt.ylabel('Ride Count')
plt.legend()
plt.show()

# Dimensionality Reduction
from pyspark.ml.feature import PCA

# PCA to use most imp feature rather than using all feature
# Principal Component Analysis(PCA)
pca = PCA(
    k=8,
    inputCol='features',
    outputCol='pcaFeatures'
)

# perform dimensionality Reduction on training data
pcaTransformer = pca.fit(vectorDF)

pcaFeatureData = pcaTransformer.transform(vectorDF).select('pcaFeatures')
pcaFeatureData.show(5)

# Each feature is Dense Vector for 8 feature
pcaFeatureData.toPandas()['pcaFeatures'][0]

# Gives idea how much every principal component contributes to the  entire dataset
print(pcaTransformer.explainedVariance)

plt.figure(figsize=(15, 6))
plt.plot(pcaTransformer.explainedVariance)
plt.xlabel('Dimension')
plt.ylabel('Explain Variance')
# Scree Plot
plt.show()

from pyspark.sql.functions import monotonically_increasing_id

# to perform join opperation on principal component and original data component which contain bike rental count
pcaFeatureData = pcaFeatureData.withColumn('row_index', monotonically_increasing_id())
vectorDF = vectorDF.withColumn('row_index', monotonically_increasing_id())

transformedData = pcaFeatureData.join(vectorDF, on=['row_index']). \
    sort('row_index'). \
    select('cnt', 'pcaFeatures')
print(transformedData.show(5))

(pcaTrainingData,pcaTestData) = transformedData.randomSplit([0.8,0.2])
pcalr = LinearRegression(maxIter=100, regParam=1.0, elasticNetParam=0.8, labelCol='cnt', featuresCol='pcaFeatures')
pcamodel = pcalr.fit(pcaTrainingData)
print('Training R^2 score = ', pcamodel.summary.r2)
print('Training RMSE ', pcamodel.summary.rootMeanSquaredError)
pcapredictions = pcamodel.transform(pcaTestData)
# predictions.show(5)

from pyspark.ml.evaluation import RegressionEvaluator

pcaevaluator = RegressionEvaluator(labelCol='cnt', predictionCol='prediction', metricName='r2')
rsquare = pcaevaluator.evaluate(pcapredictions)
print("Test R^2 score =%g" % rsquare)

evaluator = RegressionEvaluator(labelCol='cnt', predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(pcapredictions)
print("Test RMSE = ", rmse)
