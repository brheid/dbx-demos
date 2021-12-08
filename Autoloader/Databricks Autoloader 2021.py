# Databricks notebook source
# MAGIC %md ### Setup
# MAGIC Click "Run All" in the the companion `streamfiles.py` notebook in another browser tab right before running this notebook. `streamfiles.py` kicks off writes to the target directory every several seconds that we will use to demonstrate Auto Loader.

# COMMAND ----------

# clean up the workspace
dbutils.fs.rm("/tmp/iot_stream/", recurse=True)
dbutils.fs.rm("/tmp/iot_stream_chkpts/", recurse=True)
spark.sql(f"DROP TABLE IF EXISTS brheid.iot_stream")
spark.sql(f"DROP TABLE IF EXISTS brheid.iot_devices")
spark.sql("CREATE TABLE brheid.iot_devices USING DELTA AS SELECT * FROM json.`/databricks-datasets/iot/` WHERE 2=1")
spark.sql("SET spark.databricks.cloudFiles.schemaInference.enabled=true")
dbutils.fs.cp("/databricks-datasets/iot-stream/data-device/part-00000.json.gz",
              "/tmp/iot_stream/part-00000.json.gz", recurse=True)

# COMMAND ----------

input_data_path = "/tmp/iot_stream/"
input_schema = "/tmp/iot_stream/schema/"
chkpt_path = "/tmp/iot_stream_chkpts/"

# COMMAND ----------

# MAGIC %md ### <img src="https://pages.databricks.com/rs/094-YMS-629/images/dbsquare.png" width=30/> Getting your data into Delta Lake with Auto Loader and COPY INTO
# MAGIC Incrementally and efficiently load new data files into Delta Lake tables as soon as they arrive in your data lake (S3/Azure Data Lake/Google Cloud Storage).
# MAGIC 
# MAGIC <!-- <img src="https://databricks.com/wp-content/uploads/2021/02/telco-accel-blog-2-new.png" width=800/> -->
# MAGIC <img src="https://pages.databricks.com/rs/094-YMS-629/images/delta-data-ingestion.png" width=1000/>
# MAGIC 
# MAGIC <!-- <img src="https://databricks.com/wp-content/uploads/2020/02/dl-workflow2.png" width=750/> -->

# COMMAND ----------

# MAGIC %md ### <img src="https://pages.databricks.com/rs/094-YMS-629/images/dbsquare.png" width=30/> Auto Loader
# MAGIC 
# MAGIC Stream mode: Stays running and processes files as a stream as they arrive.

# COMMAND ----------

df = (spark.readStream.format("cloudFiles")
      .option("cloudFiles.format", "json")
      .option("cloudFiles.schemaLocation", input_schema) 
      .load(input_data_path))

(df.writeStream.format("delta")
 .option("checkpointLocation", chkpt_path)
   .table("brheid.iot_stream"))

# COMMAND ----------

display(df.selectExpr("COUNT(*) AS record_count"))

# COMMAND ----------

# MAGIC %md ### <img src="https://pages.databricks.com/rs/094-YMS-629/images/dbsquare.png" width=30/> Auto Loader with `triggerOnce`
# MAGIC 
# MAGIC we still keep track of which files we've processed, but the trigger once function only processes the new files when we run.
# MAGIC This job can then be scheduled and we process the new files on each execution.

# COMMAND ----------

df = (spark.readStream.format("cloudFiles")
      .option("cloudFiles.format", "json")
      .option("cloudFiles.schemaLocation", input_schema) 
      .load(input_data_path))

(df.writeStream.format("delta")
   .trigger(once=True)
   .option("checkpointLocation", chkpt_path)
   .table("brheid.iot_stream"))

# COMMAND ----------

# MAGIC %sql SELECT COUNT(*) FROM brheid.iot_stream

# COMMAND ----------

# MAGIC %md ### <img src="https://pages.databricks.com/rs/094-YMS-629/images/dbsquare.png" width=30/> SQL `COPY INTO` command
# MAGIC 
# MAGIC Instead of python cloud files, copy into is the SQL batch way to levarage autoloader.
# MAGIC It ignores data that has already been processed just like autoloader in trigger once mode.
# MAGIC 
# MAGIC Retriable, idempotent, simple.

# COMMAND ----------

# MAGIC %sql
# MAGIC COPY INTO brheid.iot_devices
# MAGIC FROM "/databricks-datasets/iot/"
# MAGIC FILEFORMAT = JSON

# COMMAND ----------

# MAGIC %sql SELECT COUNT(*) FROM brheid.iot_devices

# COMMAND ----------

# MAGIC %sql DESCRIBE HISTORY brheid.iot_devices

# COMMAND ----------

# MAGIC %md #### <img src="https://pages.databricks.com/rs/094-YMS-629/images/dbsquare.png" width=30/> View the documentation for [Auto Loader](https://docs.databricks.com/spark/latest/structured-streaming/auto-loader.html) and [COPY INTO](https://docs.databricks.com/spark/2.x/spark-sql/language-manual/copy-into.html).

# COMMAND ----------

# clean up workspace
dbutils.fs.rm("/tmp/iot_stream/", recurse=True)
dbutils.fs.rm("/tmp/iot_stream_chkpts/", recurse=True)
spark.sql(f"DROP TABLE IF EXISTS brheid.iot_stream")
spark.sql(f"DROP TABLE IF EXISTS brheid.iot_devices")
