from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
import happybase

# Step 1: Create a Spark session
spark = SparkSession.builder.appName("studentperformance").enableHiveSupport().getOrCreate()

# Step 2: Load the data from the Hive table 'performance' into a Spark DataFrame
df = spark.sql("""
        SELECT 
            CAST(Hours_Studied AS INT)                    AS Hours_Studied,
            CAST(Previous_Scores AS INT)                  AS Previous_Scores,
            CAST(Extracurricular_Activities AS STRING)    AS Extracurricular_Activities,
            CAST(Sleep_Hours AS INT)                      AS Sleep_Hours,
            CAST(Sample_Question_Papers_Practiced AS INT) AS Sample_Question_Papers_Practiced,
            CAST(Performance_Index AS INT)                AS Performance_Index
        FROM performance
    """)

# Step 3: Handle null values by either dropping or filling them
df = df.na.drop()  # Drop rows with null values

#Step 3b: Convert the string feature to numeric using StringIndexer
indexer = StringIndexer(
        inputCol="Extracurricular_Activities",
        outputCol="Extracurricular_Activities_Idx",
        handleInvalid="keep"
    )
indexed_df = indexer.fit(df).transform(df)
# Step 4: Prepare the data for MLlib by assembling features into a vector
assembler = VectorAssembler(
        inputCols=[
            "Hours_Studied",
            "Previous_Scores",
            "Extracurricular_Activities_Idx",  # numeric version of the string column
            "Sleep_Hours",
            "Sample_Question_Papers_Practiced"
        ],
        outputCol="features",
        handleInvalid="skip"
    )
assembled_df = assembler.transform(indexed_df).select("features", "Performance_Index")


# Step 5: Split the data into training and testing sets
train_data, test_data = assembled_df.randomSplit([0.7, 0.3])

# Step 6: Initialize and train a Linear Regression model
lr = LinearRegression(labelCol="Performance_Index", featuresCol="features")
lr_model = lr.fit(train_data)

# Step 7: Evaluate the model on the test data
test_results = lr_model.evaluate(test_data)

# Step 8: Print the model performance metrics
print(f"RMSE: {test_results.rootMeanSquaredError}")
print(f"R^2: {test_results.r2}")

# ---- Write metrics to HBase with happybase (using the provided pattern) ----
# Example data (row_key, column_family:column, value) populated with the metrics
data = [
    ('metrics1', 'cf:rmse', str(test_results.rootMeanSquaredError)),
    ('metrics1', 'cf:r2',   str(test_results.r2)),
]

# Function to write data to HBase inside each partition
def write_to_hbase_partition(partition):
    connection = happybase.Connection('master')
    connection.open()
    table = connection.table('my_table')  # Update table name
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()

# Parallelize data and apply the function with foreachPartition
rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Step 9: Stop the Spark session
spark.stop()




