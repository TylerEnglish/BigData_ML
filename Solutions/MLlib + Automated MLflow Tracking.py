# Databricks notebook source
# MAGIC %md
# MAGIC # Automated MLflow tracking in MLlib
# MAGIC 
# MAGIC MLflow provides automated tracking for model tuning with MLlib. With automated MLflow tracking, when you run tuning code using `CrossValidator` or `TrainValidationSplit`, the specified hyperparameters and evaluation metrics are automatically logged, making it easy to identify the optimal model.
# MAGIC 
# MAGIC This notebook shows an example of automated MLflow tracking with MLlib. 
# MAGIC 
# MAGIC This notebook uses the PySpark classes `DecisionTreeClassifier` and `CrossValidator` to train and tune a model. MLflow automatically tracks the learning process, saving the results of each run so you can examine the hyperparameters to understand the impact of each one on the model's performance and find the optimal settings.
# MAGIC 
# MAGIC This notebook uses the MNIST handwritten digit recognition dataset, which is included with Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the training and test datasets
# MAGIC 
# MAGIC The dataset is already divided into training and test sets. Each dataset has two columns: an image, represented as a vector of 784 pixels, and a "label", or the actual number shown in the image.
# MAGIC 
# MAGIC The datasets are stored in the LIBSVM dataset format.  Load them using the MLlib LIBSVM dataset reader utility.

# COMMAND ----------

# Install mlflow
%pip install mlflow

# COMMAND ----------

training = spark.read.format("libsvm") \
  .option("numFeatures", "784") \
  .load("/databricks-datasets/mnist-digits/data-001/mnist-digits-train.txt")
test = spark.read.format("libsvm") \
  .option("numFeatures", "784") \
  .load("/databricks-datasets/mnist-digits/data-001/mnist-digits-test.txt")

training.cache()
test.cache()

print("There are {} training images and {} test images.".format(training.count(), test.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC Display the data.  Each image has the true label (the `label` column) and a vector of `features` that represent pixel intensities.

# COMMAND ----------

display(training)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the ML pipeline 
# MAGIC 
# MAGIC In this example, as with most ML applications, you must do some preprocessing of the data before you can use the data to train a model. MLlib provides **pipelines** that allow you to combine multiple steps into a single workflow. In this example, you build a two-step pipeline:
# MAGIC 1. `StringIndexer` converts the labels from numeric values to categorical values. 
# MAGIC 2. `DecisionTreeClassifier` trains a decision tree that can predict the "label" column based on the data in the "features" column.
# MAGIC 
# MAGIC For more information:  
# MAGIC [Pipelines](http://spark.apache.org/docs/latest/ml-pipeline.html#ml-pipelines)

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
import mlflow
mlflow.pyspark.ml.autolog()

# COMMAND ----------

# StringIndexer: Convert the input column "label" (digits) to categorical values
indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
# DecisionTreeClassifier: Learn to predict column "indexedLabel" using the "features" column
dtc = DecisionTreeClassifier(labelCol="indexedLabel")
# Chain indexer + dtc together into a single ML Pipeline
pipeline = Pipeline(stages=[indexer, dtc])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the cross-validation 
# MAGIC 
# MAGIC Now that you have defined the pipeline, you can run the cross validation to tune the model's hyperparameters. During this process, MLflow automatically tracks the models produced by `CrossValidator`, along with their evaluation metrics. This allows you to investigate how specific hyperparameters affect the model's performance.
# MAGIC 
# MAGIC In this example, you examine two hyperparameters in the cross-validation:
# MAGIC 
# MAGIC * `maxDepth`. This parameter determines how deep, and thus how large, the tree can grow. 
# MAGIC * `maxBins`. For efficient distributed training of Decision Trees, MLlib discretizes (or "bins") continuous features into a finite number of values. The number of bins is controlled by `maxBins`. In this example, the number of bins corresponds to the number of grayscale levels; `maxBins=2` turns the images into black and white images.
# MAGIC 
# MAGIC For more information:  
# MAGIC [maxBins](https://spark.apache.org/docs/latest/mllib-decision-tree.html#split-candidates)  
# MAGIC [maxDepth](https://spark.apache.org/docs/latest/mllib-decision-tree.html#stopping-rule)

# COMMAND ----------

# Create an evaluator.  In this case, use "weightedPrecision".
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", metricName="weightedPrecision")

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

# Define the parameter grid to examine.
grid = (ParamGridBuilder() #\
  .addGrid(dtc.maxDepth, [2, 3, 4, 5, 6, 7, 8]) \
  .addGrid(dtc.maxBins, [2, 4, 8]) \
  .build())

# COMMAND ----------

# Create a cross validator, using the pipeline, evaluator, and parameter grid you created in previous steps.
cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=grid, numFolds=3)

# COMMAND ----------

# MAGIC %md
# MAGIC Run `CrossValidator`.  If an MLflow tracking server is available, `CrossValidator` automatically logs each run to MLflow, along with the evaluation metric calculated on the held-out data, under the current active run. If no run is active, a new one is created.

# COMMAND ----------

# Explicitly create a new run.
# This allows this cell to be run multiple times.
# If you omit mlflow.start_run(), then this cell could run once, but a second run would hit conflicts when attempting to overwrite the first run.
import mlflow
import mlflow.spark

with mlflow.start_run():
  # Run the cross validation on the training dataset. The cv.fit() call returns the best model it found.
  cvModel = cv.fit(training)
  
  # Evaluate the best model's performance on the test dataset and log the result.
  test_metric = evaluator.evaluate(cvModel.transform(test))
  mlflow.log_metric('test_' + evaluator.getMetricName(), test_metric) 
  
  # Log the best model.
  mlflow.spark.log_model(spark_model=cvModel.bestModel, artifact_path='best-model') 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review the logged results

# COMMAND ----------

# MAGIC %md
# MAGIC To view the MLflow experiment associated with the notebook, click the **Experiment** icon in the notebook context bar on the upper right. All notebook runs appear in the sidebar. 
# MAGIC To more easily compare their results, click the icon at the far right of **Experiment Runs** (it shows "View Experiment UI" when you hover over it). The Experiment page appears.
# MAGIC 
# MAGIC For example, to examine the effect of tuning `maxDepth`:
# MAGIC 
# MAGIC 1. On the Experiment page, enter `params.maxBins = "8"` in the **Search Runs** box, and click **Search**.
# MAGIC 1. Select the resulting runs and click **Compare**.
# MAGIC 1. In the Scatter Plot, select X-axis **maxDepth** and Y-axis **avg_weightedPrecision**.
# MAGIC 
# MAGIC You can see that, when `maxBins` is held constant at 8, the average weighted precision increases with `maxDepth`.

# COMMAND ----------

# MAGIC %md
# MAGIC # Use The Best Model to Make Predictions
# MAGIC 
# MAGIC From the "View Experiment UI" page, you can click on the most recent set of runs. There is a code block below that which tells you exactly how to recall the best model from that set of runs.

# COMMAND ----------

import mlflow
logged_model = 'runs:/916d6cdc92314982801b1821043c126c/best-model'

# Load model
loaded_model = mlflow.spark.load_model(logged_model)

# COMMAND ----------

data = test

# Perform inference via model.transform()
result = loaded_model.transform(data)

# COMMAND ----------

display(result["label", "indexedLabel", "prediction", "rawPrediction", "probability"])

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
raw_test = test.collect()

# COMMAND ----------

# Define the sparse vector
#             [item number][index of vector]
sparse_vector = raw_test[5][1]

# Convert the sparse vector to a 2D array
image = sparse_vector.toArray().reshape(28, 28)

# Display the image
plt.imshow(image, cmap='gray')
plt.show()

# COMMAND ----------


