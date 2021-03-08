# Databricks notebook source
# MAGIC %md
# MAGIC ![Databricks for Customer Churn](https://www.superoffice.com/blog/wp-content/uploads/2015/05/reduce-customer-churn.png) 
# MAGIC 
# MAGIC # Customer Churn
# MAGIC 
# MAGIC [**Customer Churn**](https://en.wikipedia.org/wiki/Customer_attrition) also known as Customer attrition, customer turnover, or customer defection, is the loss of clients or customers.
# MAGIC 
# MAGIC This demo demonstrates a simple churn analysis workflow. This explores the data set, visualizes some attributes of it, and then uses **gradient boosted trees** in Spark MLlib to try to predict churn. 
# MAGIC 
# MAGIC It then applies survival analysis, which studies the expected duration of time until an event happens. This time is often associated with some risk factors or treatment taken on the subject. One main task is to learn the relationship quantitatively and make prediction on future subjects.
# MAGIC This notebooks uses SparkR and Accelerated Failure Time regression (AFT) in Spark MLlib as well to predict _when_ customers churn. 
# MAGIC 
# MAGIC This uses the Customer Churn dataset previously in the UCI repository, but seemingly only online now at https://github.com/topepo/C5.0/blob/master/data/churn.RData for example. 

# COMMAND ----------

# %fs mv /FileStore/shared_uploads/brad.heide@databricks.com/churn_analysis.csv /Users/brad.heide@databricks.com/ml/churn/churn_analysis.csv

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *


# Read data and convert yes/no columns to boolean
df = spark.read \
  .option("inferSchema", "true") \
  .option("header", "true") \
  .csv("dbfs:/Users/brad.heide@databricks.com/ml/churn") \
  .withColumn("international_plan", col("international_plan") == "yes") \
  .withColumn("voice_mail_plan", col("voice_mail_plan") == "yes") \
  .withColumn("churn", col("churn") == "yes") \
  .cache()

# COMMAND ----------

display(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# We create a temporary view to create the delta table from 
df.createOrReplaceTempView("churn_data")

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the Delta table

# COMMAND ----------

# Configure Delta Lake Silver Path
DELTALAKE_SILVER_PATH = "dbfs:/Users/brad.heide@databricks.com/ml/churn_silver/"

# Remove folder if it exists
dbutils.fs.rm(DELTALAKE_SILVER_PATH, recurse=True)

# COMMAND ----------

# MAGIC %sql 
# MAGIC --create database brheid;
# MAGIC use brheid

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- We create the loan_by_state_delta table using delta sql syntax
# MAGIC DROP TABLE IF EXISTS churn_delta;
# MAGIC 
# MAGIC CREATE TABLE churn_delta
# MAGIC USING delta
# MAGIC LOCATION '/Users/brad.heide@databricks.com/ml/churn_silver/'
# MAGIC AS SELECT * FROM churn_data;
# MAGIC 
# MAGIC -- View Delta Lake table
# MAGIC SELECT * FROM churn_delta

# COMMAND ----------

# MAGIC %sql 
# MAGIC DESCRIBE DETAIL delta.`/Users/brad.heide@databricks.com/ml/churn_silver/`

# COMMAND ----------

# MAGIC %md ### Explore Churn Data 
# MAGIC 
# MAGIC We'll use built-in Databricks plotting, and matplotlib, to examine churn by state, and account length distribution

# COMMAND ----------

# We count the number of churn vs no churn cases
numCases = df.count()
numChurned = df.filter(col("churn")).count()
print("{} cases, {} churned, {} unchurned".format(numCases, numChurned, numCases - numChurned))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT state, count(*) AS statewise_churn FROM churn_data WHERE churn GROUP BY state ORDER BY statewise_churn DESC

# COMMAND ----------

# Same, but with matplotlib
import matplotlib.pyplot as plt
import numpy as np

importanceDF = sqlContext.sql("SELECT state, count(*) AS statewise_churn FROM churn_data WHERE churn GROUP BY state ORDER BY statewise_churn DESC").toPandas()
states = importanceDF["state"]
y_pos = np.arange(len(states))

plt.gcf().clear()
plt.bar(y_pos, importanceDF["statewise_churn"])
plt.xticks(y_pos, states)
plt.ylabel("Churned Cases")
plt.title("Churn by State")
plt.gcf().set_size_inches(20, 10)
plt.show()
display()

# COMMAND ----------

# Show distribution of account length
display(df.select("account_length"))

# COMMAND ----------

# MAGIC %md
# MAGIC From here we may find the data is skewed and we want to see how records with an account length above 220 affect our model. To this purpose we remove this data from our dataset. 

# COMMAND ----------

# MAGIC %sql
# MAGIC DELETE FROM churn_delta WHERE account_length > 220;

# COMMAND ----------

# MAGIC %sql 
# MAGIC --THis shows that the delete we did has manifested itself and is timestamped
# MAGIC DESCRIBE HISTORY churn_delta

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preparing data for experimentation

# COMMAND ----------

# We take the two versions of the data we want to experiment with
dfv0 = spark.sql("select * from churn_delta version as of 0")
dfv1 = spark.sql("select * from churn_delta version as of 1")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setting up the mlFlow experiment

# COMMAND ----------

import mlflow
print(mlflow.__version__)
import mlflow.spark

# We can either log spark models using the below option:
#spark.conf.set("spark.databricks.mlflow.trackMLlib.enabled", "true")

# Or we create and point to a specific experiment
#mlflow.create_experiment("/Users/marijse.vandenberg@databricks.com/Demo/churn_demo", artifact_location=None)
# mlflow.set_experiment("/Users/marijse.vandenberg@databricks.com/Demo/churn_demo")


# COMMAND ----------

# MAGIC %md ### Model Fitting and Summarization

# COMMAND ----------

def plot_confusion_matrix(cm):
  import matplotlib.pyplot as plt
  import numpy as np
  import itertools
  
  #cm = metrics.confusionMatrix().toArray()
  
  plt.gcf().clear()
  plt.figure()
  classes=list([0,1])
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=0)
  plt.yticks(tick_marks, classes)
  
  thresh = cm.max() / 2.0
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], '.2f'),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")
  plt.tight_layout()
  plt.ylabel('Actually Churned?')
  plt.xlabel('Predicted to Churn?')
  plt.show()
  
  # Display images
  image = plt
    
  # Save figure
  image.savefig("churn_confusion_matrix.png")
    
  # Return images
  return image

# COMMAND ----------

def data_preparation(df):
  from pyspark.ml.feature import VectorAssembler
  
  # The boolean churn col needs to be 0/1 for GBT
  
  finaldf = df.withColumn("churnedIndex", when(col("churn"), 1.0).otherwise(0.0))
  
  # Split into train and test dataset
  
  train, test = finaldf.randomSplit([0.9, 0.1], seed=12345)
  
  vecAssembler = VectorAssembler() \
    .setInputCols(["account_length", "total_day_calls",  "total_eve_calls", "total_night_calls", "total_intl_calls",  "number_customer_service_calls"]) \
    .setOutputCol("features")
  
  return vecAssembler, train, test

# COMMAND ----------

def predict_churn(df, version, maxDepth):
  from pyspark.ml import Pipeline
  from pyspark.ml.classification import GBTClassifier
  from pyspark.ml.feature import StringIndexer
  from pyspark.ml.feature import VectorAssembler
  from pyspark.mllib.evaluation import MulticlassMetrics
  
  # Prepare the data
  vecAssembler, train, test = data_preparation(df)
  
  # Start MlFlow run. 
  with mlflow.start_run():    
    # We create the model object
    classifier = GBTClassifier(maxDepth=maxDepth).setLabelCol("churnedIndex")
    
    # Now we'll tell the pipeline to first create the feature vector, and then train the classifier
    lrPipeline = Pipeline().setStages([vecAssembler, classifier])
    lrPipelineModel = lrPipeline.fit(train)  
  
    # Make predicitons from the model
    predictionsAndLabelsDF = lrPipelineModel.transform(test)
  
    # Evaluate its accuracy via the confusion matrix
    confusionMatrix = predictionsAndLabelsDF.select("prediction", "churnedIndex")
    metrics = MulticlassMetrics(confusionMatrix.rdd)
    
    cm = metrics.confusionMatrix().toArray()
    confusion_matrix = plot_confusion_matrix(cm)

    print("Overall accuracy: {}".format(metrics.accuracy))
    # Instances incorrectly predicted to churn, as a fraction of all cases that didn't churn
    # TP: correctly predicting a customer to churn
    # TN: correctly predicting a customer to not churn
    # FP: Predicting a customer to churn when it doesn't churn
    # FN: Predicting a customer to not churn when it does
    print("Fraction of non-churned cases incorrectly predicted to churn: {}".format(metrics.falsePositiveRate(1.0))) 
    print("Fraction of churned cases incorrectly predicted to not churn: {}".format(metrics.falsePositiveRate(0.0)))
    
    # Log ml flow attributes for mlflow UI - these lines are the variables that get caputured.  AUTOML will automate this!
    
    mlflow.log_param("max_depth", maxDepth)
    mlflow.log_param("data_version", version)
    
    mlflow.log_metric("accuracy", metrics.accuracy)
    mlflow.log_metric("FP / non-churned", metrics.falsePositiveRate(1.0))
    mlflow.log_metric("FN / churned", metrics.falsePositiveRate(0.0))
      
    mlflow.spark.log_model(lrPipelineModel,"GBT_model")
    mlflow.log_artifact("churn_confusion_matrix.png")
    
    modelpath = "/dbfs/Users/brad.heide@databricks.com/ml/churn_models/churn_model_%f_%s" % (maxDepth, version)
    mlflow.spark.save_model(lrPipelineModel,modelpath)
    
    #run_id = mlflow.active_run().info.run_id

    return cm

# COMMAND ----------

# We call the predict churn method on the two versions of our data, these two runs create the experiements.

cm_v0 = predict_churn(dfv0, 'v0',12)
print("----------------------------")
cm_v1 = predict_churn(dfv1, 'v1', 5)

# COMMAND ----------

# We display the confusion matrix of version 0
confusion_matrix = plot_confusion_matrix(cm_v0)
display(confusion_matrix.show())

# COMMAND ----------

# We display the confusion matrix of version 1
confusion_matrix = plot_confusion_matrix(cm_v1)
display(confusion_matrix.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score with spark dataframe

# COMMAND ----------

# once you have the model artifact you can deploy a number of ways...  get this from the experiment

#AWS okta demo shard
loaded_model = mlflow.spark.load_model("runs:/db219e4f610143ab8ce99583e634b92a/GBT_model")

#Azure FE West shard
#loaded_model = mlflow.spark.load_model("runs:/b2fa1917058c4b8596b83325f8bc7bed/GBT_model")

to_score_df = dfv0.withColumn("churnedIndex", when(col("churn"), 1.0).otherwise(0.0)).drop(col('churnedIndex'))
scored_df = loaded_model.transform(to_score_df)
display(scored_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Score with pandas dataframe

# COMMAND ----------

import mlflow

#AWS okta demo shard
logged_model = 'dbfs:/databricks/mlflow-tracking/9487869/db219e4f610143ab8ce99583e634b92a/artifacts/GBT_model'

#Azure FEWest Shard
#logged_model = 'dbfs:/databricks/mlflow-tracking/1793510902329417/b2fa1917058c4b8596b83325f8bc7bed/artifacts/GBT_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(dfv0.toPandas()))

# COMMAND ----------

# MAGIC %md # Deploy Model: Batch Scoring

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Production Model 

# COMMAND ----------

from mlflow.tracking import MlflowClient
import mlflow
import mlflow.spark
# Delete versions 1,2, and 3 of the model
client = MlflowClient()
# versions=[1, 2, 3]

# for version in versions:
#     client.delete_model_version(name="Churn_analysis", version=version)
model_name = "brheid_churn"

#Change Prod to Staging etc this pulls from the repository
prod_model = client.get_latest_versions(model_name, stages = ['Staging'])[0]  

model_source = prod_model.source
print(prod_model.name)
print("Model version: ", prod_model.version)
print("Model source: ", model_source)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score with dataframe

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F

# Load model  set above as production model
loaded_model = mlflow.spark.load_model(model_source)

# Get Input Data 
to_score_df = dfv0.withColumn("churnedIndex", when(col("churn"), 1.0).otherwise(0.0)).drop(col('churnedIndex'))
# Make Predictions
scored_df = loaded_model.transform(to_score_df)
# Add model info to each run 
results_df = (scored_df.withColumn("model_name", lit(prod_model.name))
                            .withColumn("model_version", lit(prod_model.version))
                            .withColumn("model_source", lit(model_source))
                            .withColumn("model_run_timestamp", lit(F.current_timestamp()))
             )

#Persist Results
results_df.write.mode("overwrite").format("delta").saveAsTable("brheid.churn_model_scoring")
display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explore Experiments

# COMMAND ----------

#Azure FEWest Shard
#experimentID = '1793510902329417'
# AWS
experimentID = '9487869'

# COMMAND ----------

# You can retrieve data from the mlflow ui using the following spark read syntax - put the runs into a table to track metrics over time (not ootb, but have data and can do it)
runs = spark.read.format("mlflow-experiment").load(experimentID)
display(runs)

# COMMAND ----------

# You can tidy the dataframe, for instance using below code
runs = runs.withColumn('metrics_acc', runs.metrics.accuracy)

runs = runs.withColumn('data_version', runs.params.data_version)
runs = runs.withColumn('max_depth', runs.params.max_depth)

runs = runs.drop('metrics').drop('params').drop('tags').drop('artifact_uri')

display(runs)

# COMMAND ----------

max_acc = runs.agg({"metrics_acc": "max"}).collect()[0][0]

# COMMAND ----------

from pyspark.sql.functions import col
run_id = runs.filter(col('metrics_acc')==max_acc).select(col('run_id')).take(1)[0][0]
print(run_id)

# COMMAND ----------

model_path = "runs:/"+run_id+"/GBT_model"
print(model_path)
loaded_model = mlflow.spark.load_model(model_path)
to_score_df = dfv0.withColumn("churnedIndex", when(col("churn"), 1.0).otherwise(0.0)).drop(col('churnedIndex'))
scored_df = loaded_model.transform(to_score_df)
display(scored_df)
