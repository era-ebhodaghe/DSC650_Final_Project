

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
#import happybase  

def main():
    # Step 1: Create a Spark session
    spark = (
        SparkSession.builder
        .appName("Student_Performance_Prediction")
        .enableHiveSupport()
        .getOrCreate()
    )

    # Step 2: Load the data from the Hive table 'performance' into a Spark DataFrame
    # Make sure the Hive table and column names MATCH the schema you uploaded (no spaces in names now)
    perf_df = spark.sql("""
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
    perf_df = perf_df.na.drop()  # Drop rows with null values

    # Step 3b: Convert the string feature to numeric using StringIndexer
    indexer = StringIndexer(
        inputCol="Extracurricular_Activities",
        outputCol="Extracurricular_Activities_Idx",
        handleInvalid="keep"
    )

    perf_indexed_df = indexer.fit(perf_df).transform(perf_df)

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

    assembled_df = assembler.transform(perf_indexed_df).select("features", "Performance_Index")

    # Step 5: Split the data into training and testing sets
    train_data, test_data = assembled_df.randomSplit([0.7, 0.3], seed=42)

    # Step 6: Train Linear Regression model
    lr = LinearRegression(
        labelCol="Performance_Index",
        featuresCol="features"
    )

    lr_model = lr.fit(train_data)

    # Step 7: Evaluate the model on the test data
    test_results = lr_model.evaluate(test_data)

    # Step 8: Print the model performance metrics
    print(f"RMSE: {test_results.rootMeanSquaredError}")
    print(f"R^2:  {test_results.r2}")

    # Optional: Write metrics to HBase
    data = [
        ('metrics1', 'cf:rmse', str(test_results.rootMeanSquaredError)),
        ('metrics1', 'cf:r2',   str(test_results.r2)),
    ]

    spark.stop()

if __name__ == "__main__":
    main()

