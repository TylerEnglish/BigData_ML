# Databricks notebook source
# MAGIC %md
# MAGIC ## Synapse ML 
# MAGIC #### Regression

# COMMAND ----------

# com.microsoft.azure:synapseml_2.12:0.10.2
# ^ add this text as a Maven coordinate


# importing libraries 

import synapse.ml
from synapse.ml.cognitive import *

import numpy as np
import pandas as pd

# COMMAND ----------

# reading in the data

dataFile = "AdultCensusIncome.csv"
import os, urllib
if not os.path.isfile(dataFile):
    urllib.request.urlretrieve("https://mmlspark.azureedge.net/datasets/" + dataFile, dataFile)
data = spark.createDataFrame(pd.read_csv(dataFile, dtype={" hours-per-week": np.float64}))
data.display()

# COMMAND ----------

# selecting features and splitting the data into train and test sets

data = data.select([" education", " marital-status", " hours-per-week", " income"])
train, test = data.randomSplit([0.75, 0.25], seed=123)

# COMMAND ----------

# creating the model

from synapse.ml.train import TrainClassifier
from pyspark.ml.classification import LogisticRegression
model = TrainClassifier(model=LogisticRegression(), labelCol=" income").fit(train)

# COMMAND ----------

# scoring and evaluating the model

from synapse.ml.train import ComputeModelStatistics
prediction = model.transform(test)
metrics = ComputeModelStatistics().transform(prediction)
metrics.select('accuracy').show()
