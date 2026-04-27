import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, desc, hour, to_timestamp, 
    input_file_name, substring_index, length
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import json
import pandas as pd

# 1. Initialize Spark (Optimized for Medallion Architecture)
spark = SparkSession.builder \
    .appName("YouTubeMultimodalAnalysis") \
    .master("local[*]") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# --- STEP 1: LOAD METADATA ---
print("Loading Metadata from Bronze/Raw sources...")
df_raw = spark.read.json([
    "/app/data/raw/search.json", 
    "/app/data/raw/trending.json"
])

df_metadata = df_raw.dropDuplicates(["video_id"]) \
    .filter(col("video_id").isNotNull()) \
    .withColumn("view_count", col("view_count").cast("int")) \
    .withColumn("like_count", col("like_count").cast("int")) \
    .withColumn("published_at", to_timestamp(col("published_at"))) \
    .fillna(0, subset=["view_count", "like_count"])

# --- STEP 2: LOAD BINARY THUMBNAILS ---
print("Extracting Visual Complexity from Image Binaries...")
images_raw = spark.read.format("image").load("/app/data/thumbnails/")

images_df = images_raw.select(
    substring_index(substring_index(input_file_name(), "/", -1), ".", 1).alias("img_video_id"),
    col("image.width").alias("img_width"),
    length(col("image.data")).alias("visual_complexity_bytes") 
)

# --- STEP 3: THE MULTIMODAL JOIN ---
full_df = df_metadata.join(images_df, df_metadata.video_id == images_df.img_video_id)

# --- STEP 4: ENHANCED ANALYTICS ENGINE ---
print("Running Enhanced Analytics (Diagnostic & ML)...")

# A. LOAD SILVER METRICS FOR ENHANCED DIAGNOSTIC
# We pull Brightness, Contrast, Colorfulness, and Sharpness from the Silver table
thumbnail_silver = spark.read.format("delta").load("/app/data/processed/silver/thumbnail")
enriched_df = full_df.join(thumbnail_silver, "video_id")

# B. CALCULATE CORRELATIONS
# Original "Clickbait" metric
clickbait_corr = full_df.stat.corr("visual_complexity_bytes", "view_count")
# New Engagement metrics
engagement_corr = full_df.stat.corr("view_count", "like_count")
# New Enhanced Quality metrics
quality_corr = enriched_df.stat.corr("thumbnail_quality_score", "view_count")
color_corr = enriched_df.stat.corr("colorfulness", "view_count")

# C. DESCRIPTIVE SUMMARY
descriptive_summary = full_df.groupBy("category") \
    .agg(
        count("video_id").alias("total_videos"),
        avg("view_count").alias("avg_views"),
        avg("visual_complexity_bytes").alias("avg_thumbnail_size")
    ).orderBy(desc("avg_views"))

# D. PREDICTIVE (MLlib)
assembler = VectorAssembler(
    inputCols=["view_count", "img_width"], 
    outputCol="features", 
    handleInvalid="skip"
)
ml_input = assembler.transform(full_df).select("features", "like_count")

lr = LinearRegression(featuresCol="features", labelCol="like_count")
lr_model = lr.fit(ml_input)
predictions = lr_model.transform(ml_input)

# E. PRESCRIPTIVE
best_posting_hour = full_df.withColumn("publish_hour", hour(col("published_at"))) \
    .groupBy("publish_hour") \
    .agg(avg("view_count").alias("avg_views")) \
    .orderBy(desc("avg_views"))

top_hour = best_posting_hour.first()["publish_hour"] if best_posting_hour.count() > 0 else 12

# --- STEP 5: CONSOLIDATED EXPORT ---
print("Exporting Gold Insights to JSON...")

pred_pd = predictions.limit(10).toPandas()
pred_pd['view_count_feature'] = pred_pd['features'].apply(lambda x: float(x.toArray()[0]))
predictive_json = pred_pd[['view_count_feature', 'like_count', 'prediction']].to_dict(orient="records")

final_output = {
    "level_1_descriptive": descriptive_summary.toPandas().to_dict(orient="records"),
    "level_2_diagnostic": {
        "view_like_correlation": float(engagement_corr) if engagement_corr else 0.0,
        "clickbait_thumbnail_correlation": float(clickbait_corr) if clickbait_corr else 0.0,
        "quality_view_correlation": float(quality_corr) if quality_corr else 0.0,
        "color_view_correlation": float(color_corr) if color_corr else 0.0
    },
    "level_3_predictive": predictive_json,
    "level_4_prescriptive": {
        "optimal_hour": int(top_hour),
        "action": f"Creators should upload at {int(top_hour)}:00 with vibrant, high-quality thumbnails."
    },
    "big_data_metrics": {
        "total_records_processed": full_df.count(),
        "thumbnail_data_points": images_df.count()
    }
}

with open("/app/data/final_analytics_report.json", "w") as f:
    json.dump(final_output, f, indent=4)

print("Gold Layer Complete! Data ready for Streamlit Dashboard.")
spark.stop()