from pyspark.sql import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    spark = SparkSession.builder.appName('Predicting the grape variety from wine characteristics').getOrCreate()

    rawData = spark.read.format('csv').option('header', 'false').load('../datasets/wine.data')
    print(rawData.take(5))
    print(rawData.printSchema())
    print(rawData.show(5))
    vectorizedData = vectorize(rawData)
    print("vectorizedData show",vectorizedData.show(5))
    print("vectorizedData take",vectorizedData.take(5))
    lableIndexer = StringIndexer(inputCol='label',outputCol='indexLabel')
    lableData = lableIndexer.fit(vectorizedData).transform(vectorizedData)
    print(lableData.take(5))

    print(lableData.select('label').distinct().show())
    print(lableData.select('indexLabel').distinct().show())
    (trainingData,testData) = lableData.randomSplit([0.8,0.2])
    dtree = DecisionTreeClassifier(
            labelCol='indexLabel',
            featuresCol='features',
            maxDepth=3,
            impurity='gini'
    )
    model = dtree.fit(trainingData)
    print(model)
    evaluator = MulticlassClassificationEvaluator(labelCol='indexLabel',predictionCol='prediction',metricName='f1')
    transformed_data = model.transform(testData)
    print(transformed_data.show(5))
    print(evaluator.getMetricName(),'accuracy:',evaluator.evaluate(transformed_data))

def vectorize(data):
    return data.rdd.map(lambda r: [r[0], Vectors.dense(r[1:])]).toDF(['label', 'features'])


def vectorizeAssemble(data):
     # We need to change the data into the form of two columns ("label", "features")
    assembler = VectorAssembler()
    assembler.setInputCols(data.columns[1:])
    assembler.outputCol(data.columns[0])
    return assembler.transform(data)


if __name__ == '__main__':
    main()
