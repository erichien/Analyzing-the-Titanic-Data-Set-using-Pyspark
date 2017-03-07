from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import VectorAssembler, VectorIndexer, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
import pyspark.ml
import pyspark.sql.functions as F

spark = SparkSession \
    .builder \
    .appName("titanic") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# training set

df = spark.read.csv("./train.csv", header=True, inferSchema=True)
df = df.drop('PassengerId').drop('Name').drop('Ticket').drop('Fare').drop('Cabin').drop('Embarked')
df = df.na.replace(['male', 'female'], ['1', '0'], 'Sex')
df = df.withColumn('Sex', df.Sex.cast('int'))
ave_age = df.agg(F.avg(df['age'])).collect()[0][0]
df = df.na.fill({'age': ave_age})
vecAssembler = VectorAssembler(inputCols=["Pclass", "Sex", "Age", "SibSp", "Parch"], outputCol="features")
df = vecAssembler.transform(df)
df = df.select('Survived', 'features')

df.show()
df.printSchema()

# test set

test = spark.read.csv("./test.csv", header=True, inferSchema=True)
index = test.select('PassengerId')
test = test.drop('PassengerId').drop('Name').drop('Ticket').drop('Fare').drop('Cabin').drop('Embarked')
test = test.na.replace(['male', 'female'], ['1', '0'], 'Sex')
test = test.withColumn('Sex', test.Sex.cast('int'))
ave_age = test.agg(F.avg(test['age'])).collect()[0][0]
test = test.na.fill({'age': ave_age})
vecAssembler = VectorAssembler(inputCols=["Pclass", "Sex", "Age", "SibSp", "Parch"], outputCol="features")
test = vecAssembler.transform(test)
test = test.select('features')

# Logistic Regression and Crossvalidation

lr = LogisticRegression(featuresCol="features", labelCol="Survived")
grid = ParamGridBuilder().addGrid(lr.maxIter, [10, 30, 50]).addGrid(lr.regParam, [0.01, 0.1, 1, 5]).build()
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="Survived")
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)
best_cv_Model = cv.fit(df)
f_score = best_cv_Model.avgMetrics[0]
accuracy = evaluator.evaluate(best_cv_Model.transform(df))

print 'Accuracy for Logistic Regression: ', accuracy
print 'F-Score for Logistic Regression: ', f_score

# Prediction label from test set for submission

result = best_cv_Model.transform(test)
predict_labels = result.select('prediction')
predict_labels = predict_labels.withColumn('prediction', predict_labels.prediction.cast('int'))
submit = predict_labels.withColumn('PassengerId', F.monotonically_increasing_id()+892)
submit = submit.select('PassengerId', 'prediction')
submit = submit.withColumnRenamed('prediction', 'Survived')
#submit.show()
submit.write.csv('LogisticRegression.csv', mode='overwrite', header=True)

# Random Forest

labelIndexer = StringIndexer(inputCol="Survived", outputCol="indexedLabel").fit(df)
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=3).fit(df)
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxDepth=10)
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])
grid = ParamGridBuilder().addGrid(rf.numTrees, [10, 20, 30, 40 ,50]).build()
evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")
cv = CrossValidator(estimator = pipeline, estimatorParamMaps = grid, evaluator = evaluator, numFolds = 5)
best_cv_model = cv.fit(df)
result = best_cv_model.transform(df)
f_score = best_cv_model.avgMetrics[0]
accuracy = evaluator.evaluate(best_cv_model.transform(df))

print 'Accuracy for Random Forest: ', accuracy
print 'F-Score for Random Forest: ', f_score

# Prediction label from test set for submission

result = best_cv_Model.transform(test)
predict_labels = result.select('prediction')
predict_labels = predict_labels.withColumn('prediction', predict_labels.prediction.cast('int'))
submit = predict_labels.withColumn('PassengerId', F.monotonically_increasing_id()+892)
submit = submit.select('PassengerId', 'prediction')
submit = submit.withColumnRenamed('prediction', 'Survived')
#submit.show()
submit.write.csv('RandomForest.csv', mode='overwrite', header=True)

spark.stop()