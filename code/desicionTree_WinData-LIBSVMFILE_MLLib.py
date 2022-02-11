from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import *
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.util import MLUtils


def main():
    # Configure Spark
    conf = SparkConf().setAppName("Create RDD")
    conf = conf.setMaster("local[*]").set("spark.driver.bindAddress", "localhost").set("spark.ui.port", "4050")
    spark = SparkContext(conf=conf)
    spark.setLogLevel("ERROR")

    wineRDD = MLUtils.loadLibSVMFile(spark,"../datasets/wine.scale")
    wineRDD.cache()

    print(wineRDD.take(5))


    (trainingData, testData) = wineRDD.randomSplit([0.7, 0.3])
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
    print(model.toDebugString())




if __name__ == '__main__':
    main()
