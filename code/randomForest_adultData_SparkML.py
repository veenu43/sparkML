from pyspark.sql import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import FloatType
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml import Pipeline

def main():
    spark = SparkSession.builder.appName('Predicting whether a person\'s income is greater than $50K').getOrCreate()

    rawData = spark.read.format('csv').option('header', 'false').option('ignoreLeadingWhiteSpace','true').load('../datasets/adult.csv')
    #print(rawData.take(5))
    dataset = rawData.toDF('Age',
                           'WorkClass',
                           'FnlWgt',
                           'Education',
                           'EducationNum',
                           'MaritalStatus',
                           'Occupation',
                           'Relationship',
                           'Race',
                           'Gender',
                           'CapitalGain',
                           'CapitalLoss',
                           'HoursPerWeek',
                           'NativeCountry',
                           'Label')

    #print(dataset.printSchema())
    #print("show",dataset.show(5))
    #print("Pandas Data",dataset.toPandas())
    #print("Pandas Data", dataset.toPandas().head())
    #print("count",dataset.count())
    dataset = dataset.replace('?',None)
    dataset = dataset.drop('FnlWgt')
    dataset = dataset.dropna(how='any')
    #print("count-2", dataset.count())
    #print(dataset.toPandas())
    #print(dataset.describe())
    dataset = dataset.withColumn('Age',dataset['Age'].cast(FloatType()))
    #print(dataset.describe())
    #print(dataset.toPandas())

    indexedDF = StringIndexer(inputCol='WorkClass', outputCol='WorkClass_index').fit(dataset).transform(dataset)
    #print(indexedDF.show(5))
    encodedModel = OneHotEncoder(inputCol="WorkClass_index", outputCol="WorkClass_encoded" ).fit(indexedDF)
    encodedDF = encodedModel.transform(indexedDF)
    #print(encodedDF.show(5))
    #print(encodedDF.select('WorkClass','WorkClass_index','WorkClass_encoded').toPandas().head())
    (trainingData,testData) = encodedDF.randomSplit([0.8,0.2])


    categoricalFeatures =[
        #'WorkClass',
        'Education',
        'MaritalStatus',
        'Occupation',
        'Relationship',
        'Race',
        'Gender',
        'NativeCountry'
    ]

    indexers = [StringIndexer(
        inputCol=column,
        outputCol=column+'_index',
        handleInvalid= 'keep'
    )for column in categoricalFeatures]

    encoders = [OneHotEncoder(
        inputCol=column+'_index',
        outputCol=column+'_encoded'
    ) for column in categoricalFeatures]

    lableIndexer = [(
        StringIndexer(
            inputCol='Label',
            outputCol='Label_index'
        )
    )]

    pipeline = Pipeline(stages=indexers+encoders+lableIndexer)

    transformed = pipeline.fit(trainingData).transform(trainingData)
    print(transformed.show(5,False))

    requiredFeatures = [

    ]
if __name__ == '__main__':
    main()
