import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    coalesce,
    current_timestamp,
    from_json,
    length,
    lit,
    regexp_extract,
    to_timestamp,
    unix_timestamp,
    when,
)
from pyspark.sql.types import (
    StructField,
    StructType,
    StringType,
    DoubleType,
)


class TrendingDataProcessorDelta:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.delta_bronze_path = os.path.join(self.base_dir, "data", "raw", "bronze", "trending")
        self.delta_silver_path = os.path.join(self.base_dir, "data", "processed", "silver", "trending")
        os.makedirs(os.path.dirname(self.delta_bronze_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.delta_silver_path), exist_ok=True)

    def _create_spark_session(self):
        return (
            SparkSession.builder
            .appName("YouTubeTrendingDeltaKafkaProcessing")
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
        kafka_topic = "youtube.trending.raw"

        trending_schema = StructType([
            StructField("video_id", StringType(), True),
            StructField("title", StringType(), True),
            StructField("channel_title", StringType(), True),
            StructField("published_at", StringType(), True),
            StructField("category_id", StringType(), True),
            StructField("category", StringType(), True),
            StructField("tags", StringType(), True),
            StructField("view_count", DoubleType(), True),
            StructField("like_count", DoubleType(), True),
            StructField("comment_count", DoubleType(), True),
            StructField("duration", StringType(), True),
            StructField("description", StringType(), True),
            StructField("thumbnail_url", StringType(), True),
            StructField("fetched_at", StringType(), True),
            StructField("source", StringType(), True),
        ])

        print(f"Reading trending records from Kafka topic '{kafka_topic}'...")
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
                "Publish trending records before running this processor."
            )

        parsed_df = (
            kafka_df.selectExpr("CAST(value AS STRING) AS json_str")
            .select(from_json(col("json_str"), trending_schema).alias("data"))
            .select("data.*")
        )

        print(f"Writing Bronze Delta table to {self.delta_bronze_path}...")
        (
            parsed_df.write
            .format("delta")
            .mode("append")
            .save(self.delta_bronze_path)
        )

        print("Transforming trending data to Silver layer...")
        silver_df = (
            spark.read.format("delta").load(self.delta_bronze_path)
            .dropDuplicates(["video_id"])
            .withColumn("like_count", coalesce(col("like_count").cast("double"), lit(0.0)))
            .withColumn("comment_count", coalesce(col("comment_count").cast("double"), lit(0.0)))
            .withColumn("view_count", coalesce(col("view_count").cast("double"), lit(0.0)))
            .withColumn("published_at", to_timestamp(col("published_at")))
            .withColumn("fetched_at", to_timestamp(col("fetched_at")))
            .withColumn(
                "duration_sec",
                (
                    coalesce(regexp_extract(col("duration"), r"PT(\d+)H", 1).cast("int"), lit(0)) * lit(3600)
                    + coalesce(regexp_extract(col("duration"), r"PT(?:\d+H)?(\d+)M", 1).cast("int"), lit(0)) * lit(60)
                    + coalesce(regexp_extract(col("duration"), r"PT(?:\d+H)?(?:\d+M)?(\d+)S", 1).cast("int"), lit(0))
                ).cast("int")
            )
            .withColumn(
                "engagement",
                (col("like_count") + col("comment_count"))
                / when(col("view_count") > 0, col("view_count")).otherwise(lit(1.0)),
            )
            .withColumn(
                "video_age_days",
                (unix_timestamp(current_timestamp()) - unix_timestamp(col("published_at"))) / lit(86400.0),
            )
            .withColumn("title_length", length(coalesce(col("title"), lit(""))))
            .withColumn(
                "has_thumbnail",
                when(
                    col("thumbnail_url").isNotNull()
                    & (col("thumbnail_url") != lit(""))
                    & (col("thumbnail_url") != lit("N/A")),
                    lit(True),
                ).otherwise(lit(False)),
            )
            .filter(col("view_count") > 0)
            .drop("duration")
        )

        print(f"Writing Silver Delta table to {self.delta_silver_path}...")
        (
            silver_df.write
            .format("delta")
            .mode("overwrite")
            .save(self.delta_silver_path)
        )

        print("Trending Delta pipeline complete (Kafka -> Bronze -> Silver).")
        spark.stop()


if __name__ == "__main__":
    TrendingDataProcessorDelta().process()