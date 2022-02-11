from pyspark import SparkContext, SparkConf
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics


def main():
    # Configure Spark
    conf = SparkConf().setAppName("Create RDD")
    conf = conf.setMaster("local[*]").set("spark.driver.bindAddress", "localhost").set("spark.ui.port", "4050")
    spark = SparkContext(conf=conf)
    spark.setLogLevel("ERROR")

    wineRDD = spark.textFile("../datasets/wine.data")
    wineRDD.cache()
    '''
    print(wineRDD.count())
    print(wineRDD)

    '''
    print(wineRDD.take(5))
    newRDD = wineRDD.flatMap(lambda line: line.split(","))
    print(newRDD.take(5))
    parsedRDD = wineRDD.map(parsePoint)
    print(parsedRDD.take(5))
    (trainingData, testData) = parsedRDD.randomSplit([0.7, 0.3])
    # input("Press enter to terminate")
    model = DecisionTree.trainClassifier(trainingData,
                                         numClasses=4,
                                         categoricalFeaturesInfo={},
                                         impurity='gini',
                                         maxDepth=3,
                                         maxBins=32)
    predictions = model.predict(testData.map(lambda x: x.features))
    print(predictions)
    print(predictions.take(5))
    lablesAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    print(lablesAndPredictions.take(5))
    testAcc = lablesAndPredictions.filter(lambda lp: lp[0] == lp[1]).count() / float(testData.count())
    print("Test Accuracy = ", testAcc)
    metrics = MulticlassMetrics(lablesAndPredictions)
    print(metrics.accuracy)
    print(metrics.fMeasure(1.0))
    print(metrics.precision(1.0))
    print(metrics.precision(3.0))
    print(metrics.confusionMatrix().toArray())
    print(model.toDebugString)


def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[0], values[1:])


if __name__ == '__main__':
    main()
