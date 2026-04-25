import os
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    coalesce,
    current_timestamp,
    from_json,
    length,
    lit,
    regexp_replace,
    size,
    split,
    to_timestamp,
    udf,
    when,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
)
from textblob import TextBlob
from langdetect import detect, DetectorFactory

# To ensure consistent results for language detection
DetectorFactory.seed = 0

# Define UDFs for Sentiment Analysis and Language Detection
def get_sentiment(text):
    if not text:
        return 0.0
    try:
        return float(TextBlob(text).sentiment.polarity)
    except:
        return 0.0

def detect_language(text):
    if not text or len(text) < 3:
        return "unknown"
    try:
        return detect(text)
    except:
        return "unknown"

sentiment_udf = udf(get_sentiment, DoubleType())
lang_udf = udf(detect_language, StringType())

class CommentProcessorDelta:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.delta_bronze_path = os.path.join(self.base_dir, "data", "raw", "bronze", "comments")
        self.delta_silver_path = os.path.join(self.base_dir, "data", "processed", "silver", "comments")
        os.makedirs(os.path.dirname(self.delta_bronze_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.delta_silver_path), exist_ok=True)

    def _create_spark_session(self):
        return (
            SparkSession.builder
            .appName("YouTubeCommentsDeltaKafkaProcessing")
            .master("local[*]")
            .config(
                "spark.jars.packages",
                "io.delta:delta-spark_2.12:3.0.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
            )
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .getOrCreate()
        )

    def process(self):
        spark = self._create_spark_session()
        spark.sparkContext.setLogLevel("WARN")

        kafka_bootstrap = "kafka:9092"
        kafka_topic = "youtube.comments.raw"

        comment_schema = StructType([
            StructField("video_id", StringType(), True),
            StructField("comment_id", StringType(), True),
            StructField("author_name", StringType(), True),
            StructField("comment_text", StringType(), True),
            StructField("like_count", DoubleType(), True),
            StructField("published_at", StringType(), True),
            StructField("fetched_at", StringType(), True),
            StructField("source", StringType(), True),
        ])

        print(f"Reading comment stream from Kafka topic '{kafka_topic}'...")
        try:
            kafka_df = (
                spark.read
                .format("kafka")
                .option("kafka.bootstrap.servers", kafka_bootstrap)
                .option("subscribe", kafka_topic)
                .option("startingOffsets", "earliest")
                .load()
            )
        except Exception as e:
            print(f"Error connecting to Kafka: {e}")
            spark.stop()
            return

        if kafka_df.rdd.isEmpty():
            print(f"No messages found in Kafka topic '{kafka_topic}'. Skipping Bronze update.")
            # We still might want to process Bronze if it exists
        else:
            parsed_df = (
                kafka_df.selectExpr("CAST(value AS STRING) AS json_str")
                .select(from_json(col("json_str"), comment_schema).alias("data"))
                .select("data.*")
            )

            print(f"Writing Bronze Delta table to {self.delta_bronze_path}...")
            (
                parsed_df.write
                .format("delta")
                .mode("append")
                .save(self.delta_bronze_path)
            )

        # Check if Bronze path exists before reading
        if not os.path.exists(self.delta_bronze_path):
            print("Bronze table does not exist yet. Exiting.")
            spark.stop()
            return

        print("Transforming comment data to Silver layer...")
        silver_df = (
            spark.read.format("delta").load(self.delta_bronze_path)
            .dropDuplicates(["comment_id"])
            # 1. Cleaning: Remove URLs
            .withColumn("clean_text", regexp_replace(col("comment_text"), r'http\S+|www\S+|https\S+', ''))
            # 2. Sentiment Analysis
            .withColumn("sentiment_score", sentiment_udf(col("clean_text")))
            .withColumn("sentiment_label", 
                when(col("sentiment_score") > 0.1, "positive")
                .when(col("sentiment_score") < -0.1, "negative")
                .otherwise("neutral")
            )
            # 3. Language Detection
            .withColumn("language", lang_udf(col("clean_text")))
            # 4. Metrics
            .withColumn("char_count", length(col("clean_text")))
            .withColumn("word_count", size(split(col("clean_text"), " ")))
            .withColumn("like_count", coalesce(col("like_count"), lit(0.0)))
            .withColumn("published_at", to_timestamp(col("published_at")))
            .withColumn("fetched_at", to_timestamp(col("fetched_at")))
            .withColumn("processed_at", current_timestamp())
        )

        print(f"Writing Silver Delta table to {self.delta_silver_path}...")
        (
            silver_df.write
            .format("delta")
            .mode("overwrite")
            .save(self.delta_silver_path)
        )

        print("Comment Delta pipeline complete (Kafka -> Bronze -> Silver).")
        spark.stop()

if __name__ == "__main__":
    CommentProcessorDelta().process()
