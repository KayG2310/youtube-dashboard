from itertools import count

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, desc, hour, to_timestamp
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder \
    .appName("YouTubeTrendingAnalysis") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv("/app/data/raw/trending.csv", header=True, inferSchema=True)

# --- DATA CLEANING ---
df_clean = df.withColumn("view_count", col("view_count").cast("int")) \
             .withColumn("like_count", col("like_count").cast("int")) \
             .withColumn("comment_count", col("comment_count").cast("int")) \
             .withColumn("published_at", to_timestamp(col("published_at"))) \
             .dropna(subset=["view_count", "like_count", "category"])

print("Running Descriptive Analytics")
category_summary = df_clean.groupBy("category") \
    .agg(count("video_id").alias("video_count"), 
         avg("view_count").alias("avg_views"),
         avg("like_count").alias("avg_likes")) \
    .orderBy(desc("avg_views"))

category_summary.show()

print("Running Diagnostic Analytics")
correlation = df_clean.stat.corr("view_count", "like_count")
print(f"Correlation between Views and Likes: {correlation:.2f}")

print("Running Predictive Model (MLlib)")
assembler = VectorAssembler(inputCols=["view_count"], outputCol="features")
ml_data = assembler.transform(df_clean).select("features", "like_count")

train_data, test_data = ml_data.randomSplit([0.8, 0.2], seed=42)
lr = LinearRegression(featuresCol="features", labelCol="like_count")
lr_model = lr.fit(train_data)

predictions = lr_model.transform(test_data)
predictions.select("features", "like_count", "prediction").show(5)

print("Generating Prescriptive Insights")
best_time = df_clean.withColumn("publish_hour", hour(col("published_at"))) \
    .groupBy("publish_hour") \
    .agg(avg("view_count").alias("avg_views")) \
    .orderBy(desc("avg_views"))

best_time.show(3)
recommendation = best_time.first()
print(f"PRESCRIPTIVE ACTION: Creators should post at {recommendation['publish_hour']}:00 for max views.")

# --- SAVE CSV FOR DASHBOARD ---
category_summary.toPandas().to_csv("/app/data/processed/trending_processed.csv", index=False)
category_summary.toPandas().to_json("/app/data/processed/trending_processed.json", orient="records", indent=4)
print("Pipeline complete. Processed data saved.")