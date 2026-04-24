import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    coalesce,
    current_timestamp,
    from_json,
    length,
    lit,
    to_timestamp,
    unix_timestamp,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)


class ThumbnailDataProcessorDelta:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.delta_bronze_path = os.path.join(self.base_dir, "data", "raw", "bronze", "thumbnail")
        self.delta_silver_path = os.path.join(self.base_dir, "data", "processed", "silver", "thumbnail")
        os.makedirs(os.path.dirname(self.delta_bronze_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.delta_silver_path), exist_ok=True)

    def _create_spark_session(self):
        return (
            SparkSession.builder
            .appName("YouTubeThumbnailDeltaKafkaProcessing")
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
        kafka_topic = "youtube.thumbnail.raw"

        thumbnail_schema = StructType([
            StructField("video_id", StringType(), True),
            StructField("thumbnail_url", StringType(), True),
            StructField("brightness", DoubleType(), True),
            StructField("contrast", DoubleType(), True),
            StructField("dominant_color", StringType(), True),
            StructField("colorfulness", DoubleType(), True),
            StructField("sharpness", DoubleType(), True),
            StructField("fetched_at", StringType(), True),
            StructField("source", StringType(), True),
        ])

        print(f"Reading thumbnail stream from Kafka topic '{kafka_topic}'...")
        kafka_df = (
            spark.read
            .format("kafka")
            .option("kafka.bootstrap.servers", kafka_bootstrap)
            .option("subscribe", kafka_topic)
            .option("startingOffsets", "earliest")
            .load()
        )

        if kafka_df.rdd.isEmpty():
            raise RuntimeError(
                f"No messages found in Kafka topic '{kafka_topic}'. "
                "Publish thumbnail records before running this processor."
            )

        parsed_df = (
            kafka_df.selectExpr("CAST(value AS STRING) AS json_str")
            .select(from_json(col("json_str"), thumbnail_schema).alias("data"))
            .select("data.*")
        )

        print(f"Writing Bronze Delta table to {self.delta_bronze_path}...")
        (
            parsed_df.write
            .format("delta")
            .mode("append")
            .save(self.delta_bronze_path)
        )

        print("Transforming thumbnail data to Silver layer...")
        silver_df = (
            spark.read.format("delta").load(self.delta_bronze_path)
            .dropDuplicates(["video_id"])
            .withColumn("brightness", coalesce(col("brightness"), lit(0.0)))
            .withColumn("contrast", coalesce(col("contrast"), lit(0.0)))
            .withColumn("colorfulness", coalesce(col("colorfulness"), lit(0.0)))
            .withColumn("sharpness", coalesce(col("sharpness"), lit(0.0)))
            .withColumn("fetched_at", to_timestamp(col("fetched_at")))
            .withColumn("thumbnail_quality_score", (col("brightness") + col("contrast") + col("colorfulness") + col("sharpness")) / lit(4.0))
            .filter(col("brightness") > 0)  # Basic filter for valid thumbnails
        )

        print(f"Writing Silver Delta table to {self.delta_silver_path}...")
        (
            silver_df.write
            .format("delta")
            .mode("overwrite")
            .save(self.delta_silver_path)
        )

        print("Thumbnail Delta pipeline complete (Kafka -> Bronze -> Silver).")
        spark.stop()


if __name__ == "__main__":
    ThumbnailDataProcessorDelta().process()