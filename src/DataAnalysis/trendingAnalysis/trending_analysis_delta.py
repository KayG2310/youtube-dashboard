import os
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, desc, hour, log1p, max as spark_max


def create_spark():
    return (
        SparkSession.builder
        .appName("TrendingGoldAnalysis")
        .master("local[*]")
        .config(
            "spark.jars.packages",
            "io.delta:delta-spark_2.12:3.0.0",
        )
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


def main(): 
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    silver_path = os.path.join(base_dir, "data", "processed", "silver", "trending")
    search_silver_path = os.path.join(base_dir, "data", "processed", "silver", "search")
    gold_base = os.path.join(base_dir, "data", "analysis", "gold", "trending_analysis")


    descriptive_path = os.path.join(gold_base, "descriptive")
    diagnostic_path = os.path.join(gold_base, "diagnostic")
    predictive_path = os.path.join(gold_base, "predictive")
    prescriptive_path = os.path.join(gold_base, "prescriptive")
    channel_perf_path = os.path.join(gold_base, "channel_leaderboard")
    publish_hour_path = os.path.join(gold_base, "publish_hour_effect")
    video_scoring_path = os.path.join(gold_base, "video_scoring")

    os.makedirs(gold_base, exist_ok=True)

    df = spark.read.format("delta").load(silver_path).cache()

    descriptive_df = (
        df.groupBy("category")
        .agg(
            count("video_id").alias("video_count"),
            avg("view_count").alias("avg_views"),
            avg("like_count").alias("avg_likes"),
            avg("comment_count").alias("avg_comments"),
            avg("engagement").alias("avg_engagement"),
            avg("duration_sec").alias("avg_duration_sec"),
        )
        .orderBy(desc("avg_views"))
    )
    descriptive_df.write.format("delta").mode("overwrite").save(descriptive_path)

    diagnostic_rows = [
        ("view_count_like_count", _safe_corr(df, "view_count", "like_count")),
        ("view_count_comment_count", _safe_corr(df, "view_count", "comment_count")),
        ("engagement_duration_sec", _safe_corr(df, "engagement", "duration_sec")),
        ("title_length_view_count", _safe_corr(df, "title_length", "view_count")),
    ]
    diagnostic_df = spark.createDataFrame(diagnostic_rows, ["metric_pair", "correlation"])
    diagnostic_df.write.format("delta").mode("overwrite").save(diagnostic_path)

    model_input = (
        df.select(
            "view_count",
            "like_count",
            "comment_count",
            "duration_sec",
            "video_age_days",
            "title_length",
            "category",
        )
        .dropna(subset=["view_count", "like_count", "comment_count", "duration_sec", "video_age_days", "title_length", "category"])
    )

    category_indexer = StringIndexer(inputCol="category", outputCol="category_index", handleInvalid="keep")
    indexed_df = category_indexer.fit(model_input).transform(model_input)

    assembler = VectorAssembler(
        inputCols=["view_count", "comment_count", "duration_sec", "video_age_days", "title_length", "category_index"],
        outputCol="features",
    )
    ml_df = assembler.transform(indexed_df).select("features", col("like_count").alias("label"))

    train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
    lr = LinearRegression(featuresCol="features", labelCol="label")
    model = lr.fit(train_df)
    pred_df = (
        model.transform(test_df)
        .select(
            col("label").alias("actual_like_count"),
            col("prediction").alias("predicted_like_count"),
        )
    )
    pred_df.write.format("delta").mode("overwrite").save(predictive_path)

    hourly_df = (
        df.withColumn("publish_hour", hour(col("published_at")))
        .groupBy("category", "publish_hour")
        .agg(
            avg("view_count").alias("avg_views"),
            avg("engagement").alias("avg_engagement"),
        )
        .orderBy(desc("avg_views"))
    )
    prescriptive_df = (
        hourly_df.groupBy("category")
        .agg(
            spark_max("avg_views").alias("best_avg_views"),
            spark_max("avg_engagement").alias("best_avg_engagement"),
        )
    )
    prescriptive_df.write.format("delta").mode("overwrite").save(prescriptive_path)

    channel_df = (
        df.groupBy("channel_title")
        .agg(
            count("video_id").alias("videos"),
            avg("view_count").alias("avg_views"),
            avg("engagement").alias("avg_engagement"),
            avg("like_count").alias("avg_likes"),
        )
        .orderBy(desc("avg_views"))
    )
    channel_df.write.format("delta").mode("overwrite").save(channel_perf_path)

    publish_hour_df = (
        df.withColumn("publish_hour", hour(col("published_at")))
        .groupBy("publish_hour")
        .agg(
            count("video_id").alias("video_count"),
            avg("view_count").alias("avg_views"),
            avg("engagement").alias("avg_engagement"),
        )
        .orderBy("publish_hour")
    )
    publish_hour_df.write.format("delta").mode("overwrite").save(publish_hour_path)

    video_scoring_df = (
        df.select(
            "video_id",
            "title",
            "channel_title",
            "category",
            "view_count",
            "like_count",
            "comment_count",
            "engagement",
            "video_age_days",
            "duration_sec",
            "published_at",
        )
        .withColumn(
            "score",
            (log1p(col("view_count")) * 0.45)
            + (log1p(col("like_count") + col("comment_count")) * 0.30)
            + ((col("engagement") * 100.0) * 0.20)
            - (col("video_age_days") * 0.05),
        )
        .orderBy(desc("score"))
    )
    video_scoring_df.write.format("delta").mode("overwrite").save(video_scoring_path)

    print("Trending analysis complete. Delta tables written:")
    print(f"- {descriptive_path}")
    print(f"- {diagnostic_path}")
    print(f"- {predictive_path}")
    print(f"- {prescriptive_path}")
    print(f"- {channel_perf_path}")
    print(f"- {publish_hour_path}")
    print(f"- {video_scoring_path}")

    spark.stop()


def _safe_corr(df, left, right):
    value = df.stat.corr(left, right)
    return float(value) if value is not None else 0.0


if __name__ == "__main__":
    main()