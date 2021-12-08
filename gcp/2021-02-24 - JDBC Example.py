# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook shows you how to load data from JDBC databases using Spark SQL.
# MAGIC 
# MAGIC *For production, you should control the level of parallelism used to read data from the external database, using the parameters described in the documentation.*

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 1: Connection Information
# MAGIC 
# MAGIC This is a **Python** notebook so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` magic command. Python, Scala, SQL, and R are all supported.
# MAGIC 
# MAGIC First we'll define some variables to let us programmatically create these connections.

# COMMAND ----------

driver = "org.postgresql.Driver"
url = "jdbc:postgresql://database_server"
table = "schema.tablename"
user = ""
password = ""

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 2: Reading the data
# MAGIC 
# MAGIC Now that we specified our file metadata, we can create a DataFrame. You'll notice that we use an *option* to specify that we'd like to infer the schema from the file. We can also explicitly set this to a particular schema if we have one already.
# MAGIC 
# MAGIC First, let's create a DataFrame in Python, notice how we will programmatically reference the variables we defined above.

# COMMAND ----------

remote_table = spark.read.format("jdbc")\
  .option("driver", driver)\
  .option("url", url)\
  .option("dbtable", table)\
  .option("user", user)\
  .option("password", password)\
  .load()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 3: Querying the data
# MAGIC 
# MAGIC Now that we created our DataFrame. We can query it. For instance, you can select some particular columns to select and display within Databricks.

# COMMAND ----------

display(remote_table.select("EXAMPLE_COLUMN"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 4: (Optional) Create a view or table
# MAGIC 
# MAGIC If you'd like to be able to use query this data as a table, it is simple to register it as a *view* or a table.

# COMMAND ----------

remote_table.createOrReplaceTempView("YOUR_TEMP_VIEW_NAME")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We can query this using Spark SQL. For instance, we can perform a simple aggregation. Notice how we can use `%sql` in order to query the view from SQL.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT EXAMPLE_GROUP, SUM(EXAMPLE_AGG) FROM YOUR_TEMP_VIEW_NAME GROUP BY EXAMPLE_GROUP

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Since this table is registered as a temp view, it will be available only to this notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.

# COMMAND ----------

remote_table.write.format("parquet").saveAsTable("MY_PERMANENT_TABLE_NAME")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This table will persist across cluster restarts as well as allow various users across different notebooks to query this data. However, this will not connect back to the original database when doing so.
