# Databricks notebook source
# MAGIC %md
# MAGIC <!-- You can run this notebook in a Databricks environment. Specifically, this notebook has been designed to run in [Databricks Community Edition](http://community.cloud.databricks.com/) as well. -->
# MAGIC To run this notebook, you have to [create a cluster](https://docs.databricks.com/clusters/create.html) with version **Databricks Runtime 7.4 or later** and [attach this notebook](https://docs.databricks.com/notebooks/notebooks-manage.html#attach-a-notebook-to-a-cluster) to that cluster. <br/>
# MAGIC 
# MAGIC ### Source Data for this notebook
# MAGIC The data used is a modified version of the public data from [Lending Club](https://www.kaggle.com/wendykan/lending-club-loan-data). It includes all funded loans from 2012 to 2017. Each loan includes applicant information provided by the applicant as well as the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. For a full view of the data please view the data dictionary available [here](https://resources.lendingclub.com/LCDataDictionary.xlsx).
# MAGIC 
# MAGIC Video presentation [link](https://youtu.be/BMO90DI82Dc)

# COMMAND ----------

# MAGIC %md ## Setup

# COMMAND ----------

# MAGIC %md NOTES
# MAGIC Using the downloaded parquet file in DBFS for most things, but have a copy of the parquet file in ADLS which I use up front to demonstrate the reach into ADLS.
# MAGIC Adjusted the Delta features list to correllate to order of shown features.
# MAGIC After stream visualtization, move to DBXSQL and query table. Create same visualization. Show Schema browswer and do quick tour.
# MAGIC Make sure endpoint is up and runnign or this will be a fail.

# COMMAND ----------

db = "brheid_deltadb"

spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")
spark.sql(f"USE {db}")

spark.sql("SET spark.databricks.delta.formatCheck.enabled = false")
spark.sql("SET spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite = true")

# COMMAND ----------

import random
from datetime import datetime
from pyspark.sql.functions import *
from pyspark.sql.types import *


def my_checkpoint_dir(): 
  return "/brheid/delta_demo/chkpt/%s" % str(random.randint(0, 10000))

# User-defined function to generate random state
@udf(returnType=StringType())
def random_state():
  return str(random.choice(["CA", "TX", "NY", "WA"]))


# Function to start a streaming query with a stream of randomly generated load data and append to the parquet table
def generate_and_append_data_stream(table_format, table_name, schema_ok=False, type="batch"):
  
  stream_data = (spark.readStream.format("rate").option("rowsPerSecond", 500).load()
    .withColumn("loan_id", 10000 + col("value"))
    .withColumn("funded_amnt", (rand() * 5000 + 5000).cast("integer"))
    .withColumn("paid_amnt", col("funded_amnt") - (rand() * 2000))
    .withColumn("addr_state", random_state())
    .withColumn("type", lit(type)))
    
  if schema_ok:
    stream_data = stream_data.select("loan_id", "funded_amnt", "paid_amnt", "addr_state", "type", "timestamp")
      
  query = (stream_data.writeStream
    .format(table_format)
    .option("checkpointLocation", my_checkpoint_dir())
    .trigger(processingTime = "5 seconds")
    .table(table_name))

  return query

# COMMAND ----------

# Function to stop all streaming queries 
def stop_all_streams():
    print("Stopping all streams")
    for s in spark.streams.active:
        try:
            s.stop()
        except:
            pass
    print("Stopped all streams")
    dbutils.fs.rm("/brheid/delta_demo/chkpt/", True)


def cleanup_paths_and_tables():
    dbutils.fs.rm("/brheid/delta_demo/", True)
    dbutils.fs.rm("file:/dbfs/brheid/delta_demo/loans_parquet/", True)
        
    for table in ["brheid_deltadb.loans_parquet", "brheid_deltadb.loans_delta", "brheid_deltadb.loans_delta2"]:
        spark.sql(f"DROP TABLE IF EXISTS {table}")
    
cleanup_paths_and_tables()

# COMMAND ----------

# MAGIC %sh mkdir -p /dbfs/brheid/delta_demo/loans_parquet/; wget -O /dbfs/brheid/delta_demo/loans_parquet/loans.parquet https://pages.databricks.com/rs/094-YMS-629/images/SAISEU19-loan-risks.snappy.parquet

# COMMAND ----------

# MAGIC %md <img src="https://docs.delta.io/latest/_static/delta-lake-logo.png" width=300/>

# COMMAND ----------

# MAGIC %md <img src="/files/lakehouse.png" width=1024/>

# COMMAND ----------

# MAGIC %md # What is <img src="https://docs.delta.io/latest/_static/delta-lake-logo.png" width=300/>
# MAGIC 
# MAGIC An open-source storage layer for data lakes that brings ACID transactions to Apache Spark™ and big data workloads.
# MAGIC 
# MAGIC * **Open Format**: Stored as Parquet format in blob storage.
# MAGIC * **Unified Batch and Streaming Source and Sink**: A table in Delta Lake is both a batch table, as well as a streaming source and sink. Streaming data ingest, batch historic backfill, and interactive queries all just work out of the box. 
# MAGIC * **ACID Transactions**: Ensures data integrity and read consistency with complex, concurrent data pipelines.
# MAGIC * **Audit History**: History of all the operations that happened in the table.
# MAGIC * **Schema Enforcement and Evolution**: Ensures data cleanliness by blocking writes with unexpected.
# MAGIC * **Time Travel**: Query previous versions of the table by time or version number.
# MAGIC * **Deletes and upserts**: Supports deleting and upserting into tables with programmatic APIs.
# MAGIC * **Scalable Metadata management**: Able to handle millions of files are scaling the metadata operations with Spark.

# COMMAND ----------

# MAGIC %md ## ![Delta Lake Tiny Logo](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Convert to Delta Lake format

# COMMAND ----------

# MAGIC %md Delta Lake is 100% compatible with Apache Spark&trade;, which makes it easy to get started with if you already use Spark for your big data workflows.
# MAGIC Delta Lake features APIs for **SQL**, **Python**, and **Scala**, so that you can use it in whatever language you feel most comfortable in.

# COMMAND ----------

# MAGIC %md <img src="https://databricks.com/wp-content/uploads/2020/12/simplysaydelta.png" width=600/>

# COMMAND ----------

# MAGIC %md In **Python**: Read your data into a Spark DataFrame, then write it out in Delta Lake format directly, with no upfront schema definition needed.

# COMMAND ----------

#parquet_path = "file:/dbfs/brheid/delta_demo/loans_parquet/"
parquet_path = "abfss://deltalake@oneenvadls.dfs.core.windows.net/brheid/lendingclub/"

df = (spark.read.format("parquet").load(parquet_path)
      .withColumn("type", lit("batch"))
      .withColumn("timestamp", current_timestamp()))

df.show(10)
df.write.format("delta").mode("overwrite").saveAsTable("loans_delta")

# COMMAND ----------

# MAGIC %md **SQL:** Use `CREATE TABLE` statement with SQL (no upfront schema definition needed)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE brheid_deltadb.loans_delta2
# MAGIC USING delta
# MAGIC AS SELECT * FROM parquet.`abfss://deltalake@oneenvadls.dfs.core.windows.net/brheid/lendingclub/`;
# MAGIC --`/brheid/delta_demo/loans_parquet`;
# MAGIC Select * from brheid_deltadb.loans_delta2;

# COMMAND ----------

# MAGIC %md **SQL**: Use `CONVERT TO DELTA` to convert Parquet files to Delta Lake format in place

# COMMAND ----------

# MAGIC %sql CONVERT TO DELTA parquet.`/brheid/delta_demo/loans_parquet`

# COMMAND ----------

# MAGIC %md ### View the data in the Delta Lake table
# MAGIC **How many records are there, and what does the data look like?**

# COMMAND ----------

spark.sql("select count(*) from brheid_deltadb.loans_delta").show()
spark.sql("select * from brheid_deltadb.loans_delta").show(3)

# COMMAND ----------

# MAGIC %md ## ![Delta Lake Tiny Logo](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Unified batch + streaming data processing with multiple concurrent readers and writers
# MAGIC 
# MAGIC The Lambda architecture is a big data processing architecture that combines both batch and real-time processing methods.
# MAGIC It features an append-only immutable data source that serves as system of record. Timestamped events are appended to 
# MAGIC existing events (nothing is overwritten). Data is implicitly ordered by time of arrival. 
# MAGIC 
# MAGIC Notice how there are really two pipelines here, one batch and one streaming, hence the name <i>lambda</i> architecture.
# MAGIC 
# MAGIC It is very difficult to combine processing of batch and real-time data as is evidenced by the diagram below.
# MAGIC 
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/Delta/lambda.png" style="height: 400px"/></div><br/>
# MAGIC 
# MAGIC ## Databricks solution:
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/Delta/delta.png" style="height: 350px"/></div><br/>

# COMMAND ----------

# MAGIC %md ### Write 2 different data streams into our Delta Lake table at the same time.

# COMMAND ----------

# Set up 2 streaming writes to our table
stream_query_A = generate_and_append_data_stream(table_format="delta", table_name="brheid_deltadb.loans_delta", schema_ok=True, type='stream A')
stream_query_B = generate_and_append_data_stream(table_format="delta", table_name="brheid_deltadb.loans_delta", schema_ok=True, type='stream B')

# COMMAND ----------

# MAGIC %md ### Create 2 continuous streaming readers of our Delta Lake table to illustrate streaming progress.

# COMMAND ----------

# Streaming read #1
display(spark.readStream.format("delta").table("brheid_deltadb.loans_delta").groupBy("type").count().orderBy("type"))

# COMMAND ----------

# Streaming read #2
display(spark.readStream.format("delta").table("brheid_deltadb.loans_delta").groupBy("type", window("timestamp", "10 seconds")).count().orderBy("window"))

# COMMAND ----------

# MAGIC %md ### Add a batch query to show we can serve consistent results even while the table is in use. Concurent read/Write with delta!
# MAGIC Show the same query in DBX SQL

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT addr_state, COUNT(*)
# MAGIC FROM brheid_deltadb.loans_delta
# MAGIC GROUP BY addr_state

# COMMAND ----------

dbutils.notebook.exit("stop")

# COMMAND ----------

stop_all_streams()

# COMMAND ----------

# MAGIC %md ## ![Delta Lake Tiny Logo](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) ACID Transactions
# MAGIC Audit history of all transactions for a table

# COMMAND ----------

# MAGIC %md View the Delta Lake transaction log
# MAGIC 
# MAGIC /dbfs/brheid/delta_demo/loans_parquet/_delta_log

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Show audit history of table
# MAGIC DESCRIBE HISTORY brheid_deltadb.loans_delta

# COMMAND ----------

# MAGIC %md <img src="https://databricks.com/wp-content/uploads/2020/09/delta-lake-medallion-model-scaled.jpg" width=1012/>

# COMMAND ----------

# MAGIC %md ##  ![Delta Lake Tiny Logo](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Use Schema Enforcement to protect data quality

# COMMAND ----------

# MAGIC %md To show you how schema enforcement works, let's create a new table that has an extra column -- `credit_score` -- that doesn't match our existing Delta Lake table schema.

# COMMAND ----------

# MAGIC %md #### Write DataFrame with extra column, `credit_score`, to Delta Lake table

# COMMAND ----------

# Generate `new_data` with additional column CREDIT SCORE
new_column = [StructField("credit_score", IntegerType(), True)]
new_schema = StructType(spark.table("loans_delta").schema.fields + new_column)
data = [(99997, 10000, 1338.55, "CA", "batch", datetime.now(), 649),
        (99998, 20000, 1442.55, "NY", "batch", datetime.now(), 702)]

new_data = spark.createDataFrame(data, new_schema)
new_data.printSchema()
new_data.show()

# COMMAND ----------

# Uncommenting this cell will lead to an error because the schemas don't match.
# Attempt to write data with new column to Delta Lake table
new_data.write.format("delta").mode("append").saveAsTable("brheid_deltadb.loans_delta")

# COMMAND ----------

# MAGIC %md **Schema enforcement helps keep our tables clean and tidy so that we can trust the data we have stored in Delta Lake.** The writes above were blocked because the schema of the new data did not match the schema of table (see the exception details). See more information about how it works [here](https://databricks.com/blog/2019/09/24/diving-into-delta-lake-schema-enforcement-evolution.html).

# COMMAND ----------

# MAGIC %md ##  ![Delta Lake Tiny Logo](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Use Schema Evolution to add new columns to schema
# MAGIC 
# MAGIC If we *want* to update our Delta Lake table to match this data source's schema, we can do so using schema evolution. Simply add the following to the Spark write command: `.option("mergeSchema", "true")`

# COMMAND ----------

new_data.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("brheid_deltadb.loans_delta")

# COMMAND ----------

# MAGIC %sql SELECT * FROM brheid_deltadb.loans_delta WHERE loan_id IN (99997, 99998)

# COMMAND ----------

# MAGIC %md ## ![Delta Lake Tiny Logo](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Delta Lake Time Travel

# COMMAND ----------

# MAGIC %md Delta Lake’s time travel capabilities simplify building data pipelines for use cases including:
# MAGIC 
# MAGIC * Auditing Data Changes
# MAGIC * Reproducing experiments & reports
# MAGIC * Rollbacks
# MAGIC 
# MAGIC As you write into a Delta table or directory, every operation is automatically versioned.
# MAGIC 
# MAGIC <img src="https://github.com/risan4841/img/blob/master/transactionallogs.png?raw=true" width=250/>
# MAGIC 
# MAGIC You can query snapshots of your tables by:
# MAGIC 1. **Version number**, or
# MAGIC 2. **Timestamp.**
# MAGIC 
# MAGIC using Python, Scala, and/or SQL syntax; for these examples we will use the SQL syntax.  

# COMMAND ----------

# MAGIC %md #### Review Delta Lake Table History for  Auditing & Governance
# MAGIC All the transactions for this table are stored within this table including the initial set of insertions, update, delete, merge, and inserts with schema modification

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY brheid_deltadb.loans_delta

# COMMAND ----------

# MAGIC %md #### Use time travel to select and view the original version of our table (Version 0).
# MAGIC As you can see, this version contains the original 14,705 records in it.

# COMMAND ----------

# MAGIC %sql SELECT COUNT(*) FROM brheid_deltadb.loans_delta

# COMMAND ----------

spark.sql("SELECT * FROM brheid_deltadb.loans_delta VERSION AS OF 0").show(3)
spark.sql("SELECT COUNT(*) FROM brheid_deltadb.loans_delta VERSION AS OF 0").show()

# COMMAND ----------

# MAGIC %md #### Rollback a table to a specific version using `RESTORE`

# COMMAND ----------

# MAGIC %sql RESTORE brheid_deltadb.loans_delta VERSION AS OF 0

# COMMAND ----------

# MAGIC %sql SELECT COUNT(*) FROM brheid_deltadb.loans_delta

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Delta Lake Logo Tiny](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Full DML Support: `DELETE`, `UPDATE`, `MERGE INTO`
# MAGIC 
# MAGIC Delta Lake brings ACID transactions and full DML support to data lakes.
# MAGIC 
# MAGIC >Parquet does **not** support these commands - they are unique to Delta Lake.

# COMMAND ----------

# MAGIC %md ###![Delta Lake Logo Tiny](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) `DELETE`: Handle GDPR or CCPA Requests on your Data Lake

# COMMAND ----------

# MAGIC %md Imagine that we are responding to a GDPR data deletion request. The user with loan ID #4420 wants us to delete their data. Here's how easy it is.

# COMMAND ----------

# MAGIC %md **View the user's data**

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM brheid_deltadb.loans_delta WHERE loan_id=4420

# COMMAND ----------

# MAGIC %md **Delete the individual user's data with a single `DELETE` command using Delta Lake.**
# MAGIC 
# MAGIC Note: The `DELETE` command isn't supported in Parquet.

# COMMAND ----------

# MAGIC %sql
# MAGIC DELETE FROM brheid_deltadb.loans_delta WHERE loan_id=4420;
# MAGIC -- Confirm the user's data was deleted
# MAGIC SELECT * FROM brheid_deltadb.loans_delta WHERE loan_id=4420

# COMMAND ----------

# MAGIC %md ###![Delta Lake Logo Tiny](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png)  Use time travel and `INSERT INTO` to add the user back into our table

# COMMAND ----------

# MAGIC %sql
# MAGIC -- grab the deleted row from the previous version of the table
# MAGIC 
# MAGIC INSERT INTO brheid_deltadb.loans_delta
# MAGIC SELECT * FROM brheid_deltadb.loans_delta VERSION AS OF 0
# MAGIC WHERE loan_id=4420

# COMMAND ----------

# MAGIC %sql SELECT * FROM brheid_deltadb.loans_delta WHERE loan_id=4420

# COMMAND ----------

# MAGIC %md ### ![Delta Lake Logo Tiny](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) `UPDATE`: Modify the existing records in a table in one command

# COMMAND ----------

# MAGIC %sql UPDATE brheid_deltadb.loans_delta SET funded_amnt = 21000 WHERE loan_id = 4420

# COMMAND ----------

# MAGIC %sql SELECT * FROM brheid_deltadb.loans_delta WHERE loan_id = 4420

# COMMAND ----------

# MAGIC %md ###![Delta Lake Logo Tiny](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Support Change Data Capture Workflows & Other Ingest Use Cases via `MERGE INTO`

# COMMAND ----------

# MAGIC %md
# MAGIC With a legacy data pipeline, to insert or update a table, you must:
# MAGIC 1. Identify the new rows to be inserted
# MAGIC 2. Identify the rows that will be replaced (i.e. updated)
# MAGIC 3. Identify all of the rows that are not impacted by the insert or update
# MAGIC 4. Create a new temp based on all three insert statements
# MAGIC 5. Delete the original table (and all of those associated files)
# MAGIC 6. "Rename" the temp table back to the original table name
# MAGIC 7. Drop the temp table
# MAGIC 
# MAGIC <img src="https://pages.databricks.com/rs/094-YMS-629/images/merge-into-legacy.gif" alt='Merge process' width=600/>
# MAGIC 
# MAGIC 
# MAGIC #### INSERT or UPDATE with Delta Lake
# MAGIC 
# MAGIC 2-step process: 
# MAGIC 1. Identify rows to insert or update
# MAGIC 2. Use `MERGE`

# COMMAND ----------

# Create merge table with 1 row update, 1 insertion
# chaned the funded amount back and updated paid amount.


data = [(4420, 22000, 21500.00, "NY", "update", datetime.now()),  # record to update
        (99999, 10000, 1338.55, "CA", "insert", datetime.now())]  # record to insert
schema = spark.table("brheid_deltadb.loans_delta").schema
spark.createDataFrame(data, schema).createOrReplaceTempView("merge_table")
spark.sql("SELECT * FROM merge_table").show()

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO brheid_deltadb.loans_delta AS l
# MAGIC USING merge_table AS m
# MAGIC ON l.loan_id = m.loan_id
# MAGIC WHEN MATCHED THEN 
# MAGIC   UPDATE SET *
# MAGIC WHEN NOT MATCHED 
# MAGIC   THEN INSERT *;

# COMMAND ----------

# MAGIC %sql SELECT * FROM brheid_deltadb.loans_delta WHERE loan_id IN (4420, 99999)

# COMMAND ----------

# MAGIC %md ## ![Delta Lake Logo Tiny](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) File compaction and performance optimizations = faster queries

# COMMAND ----------

# MAGIC %md ### Vacuum

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Vacuum deletes all files no longer needed by the current version of the table.
# MAGIC VACUUM brheid_deltadb.loans_delta

# COMMAND ----------

# MAGIC %md ### <img src="https://pages.databricks.com/rs/094-YMS-629/images/dbsquare.png" width=30/> Cache table in memory (Databricks Delta Lake only)

# COMMAND ----------

# MAGIC %sql CACHE SELECT * FROM brheid_deltadb.loans_delta

# COMMAND ----------

# MAGIC %md ### <img src="https://pages.databricks.com/rs/094-YMS-629/images/dbsquare.png" width=30/> Z-Order Optimize (Databricks Delta Lake only)

# COMMAND ----------

# MAGIC %sql OPTIMIZE brheid_deltadb.loans_delta ZORDER BY addr_state

# COMMAND ----------

cleanup_paths_and_tables()

# COMMAND ----------

# MAGIC %md <img src="https://docs.delta.io/latest/_static/delta-lake-logo.png" width=300/>
