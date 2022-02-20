from pyspark.sql import *
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.sql.types import FloatType

def main():
    spark = SparkSession.builder.appName('Predicting the price of an automobile given a set of features').getOrCreate()

    rawData = spark.read.format('csv').option('header', 'true').load('../datasets/imports-85.data')
    print(rawData.show(5))
    #print(rawData.toPandas().head())
    '''
    dataset = rawData.withColumn('price', rawData['price'].cast(FloatType()))
    dataset = dataset.withColumn('make', dataset['make'].cast(FloatType()))
    dataset = dataset.withColumn('num-of-doors', dataset['num-of-doors'].cast(FloatType()))
    dataset = dataset.withColumn('body-style', dataset['body-style'].cast(FloatType()))
    dataset = dataset.withColumn('wheel-base', dataset['wheel-base'].cast(FloatType()))

    dataset = dataset.withColumn('drive-wheels', dataset['drive-wheels'].cast(FloatType()))
    dataset = dataset.withColumn('curb-weight', dataset['curb-weight'].cast(FloatType()))
    dataset = dataset.withColumn('num-of-cylinders', dataset['num-of-cylinders'].cast(FloatType()))
    dataset = dataset.withColumn('engine-size', dataset['engine-size'].cast(FloatType()))
    dataset = dataset.withColumn('horsepower', dataset['horsepower'].cast(FloatType()))
    dataset = dataset.withColumn('peak-rpm', dataset['peak-rpm'].cast(FloatType()))
    '''
    dataset = rawData.select(
        col('price').cast('float'),
        col('make'),
        col('num-of-doors'),
        col('body-style'),
        col('drive-wheels'),
        col('wheel-base').cast('float'),
        col('curb-weight').cast('float'),
        col('num-of-cylinders'),
        col('engine-size').cast('float'),
        col('horsepower').cast('float'),
        col('peak-rpm').cast('float')
    )

    print(dataset.show(5))
    print(dataset.count())
    dataset = dataset.replace('?', None).dropna(how='any')
    print(dataset.count())
    (trainingData, testData) = dataset.randomSplit([0.8, 0.2])
    print(f"TrainingData : {trainingData.count()}, testData: {testData.count()}")
    categoricalfeatures = [
        'make',
        'num-of-doors',
        'body-style',
        'drive-wheels',
        'num-of-cylinders'
    ]
    indexers = [StringIndexer(
        inputCol=column,
        outputCol=column + '_index',
        handleInvalid='keep') for column in categoricalfeatures]

    encoder = [OneHotEncoder(
        inputCol=column + '_index',
        outputCol=column + '_encoded',
        handleInvalid='keep') for column in categoricalfeatures]

    requiredFeatures = [
        'make_encoded',
        'num-of-doors_encoded',
        'body-style_encoded',
        'drive-wheels_encoded',
        'wheel-base',
        'curb-weight',
        'num-of-cylinders_encoded',
        'engine-size',
        'horsepower',
        'peak-rpm'
    ]

    assembler = VectorAssembler(inputCols=requiredFeatures, outputCol='features')

    # maxIter = no of epochs for which we run the training process
    # regParam = penality applied to net co-efficient
    # elasticNetParam = Equivalent of alpha.When set to 0.0 it becomes a Lasso model.Setting to 1.0 makes it use Ridge regression
    lr = LinearRegression(maxIter=100, regParam=1.0, elasticNetParam=0.8, labelCol='price', featuresCol='features')
    pipeline = Pipeline(stages=indexers + encoder + [assembler, lr])

    # This model is pipeline model not linear model
    model = pipeline.fit(trainingData)

    # Linear model is last stage of the model
    lrmodel = model.stages[-1]

    # R^2 captures how well model captures variance in the underlying model.Higher square means better model
    print('Training R^2 score = ', lrmodel.summary.r2)

    # RMSE gives avg values by which prediction varies from actual value
    print('Training RMSE = ', lrmodel.summary.rootMeanSquaredError)

    print("No Of features", lrmodel.numFeatures)

    print("Features coefficients", lrmodel.numFeatures)

    predictions = model.transform(testData)
    print("Predictions", predictions.show(5))
    print("Predictions", predictions.toPandas().head())
    print("Predictions Features", predictions.toPandas()['features'][0])
    evaluator = RegressionEvaluator(labelCol='price',predictionCol='prediction',metricName='r2')
    r2 = evaluator.evaluate(predictions)
    print('Test R^2 score =', r2)
    evaluator = RegressionEvaluator(labelCol='price', predictionCol='prediction', metricName='rmse')
    rmse = evaluator.evaluate(predictions)
    print('Test rmse  =', rmse)
    predictionsPanadasDF = predictions.select(
        col('price'),
        col('prediction')
    ).toPandas()
    print(predictionsPanadasDF.head())

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,6))
    plt.plot(predictionsPanadasDF['price'],label='Actual')
    plt.plot(predictionsPanadasDF['prediction'], label='Predicted')
    plt.ylabel("Price")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()
