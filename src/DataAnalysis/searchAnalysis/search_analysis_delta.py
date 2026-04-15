import os
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, desc, hour, log1p


def create_spark():
    return (
        SparkSession.builder
        .appName("SearchGoldAnalysis")
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
    silver_path = os.path.join(base_dir, "data", "processed", "silver", "search")
    gold_base = os.path.join(base_dir, "data", "analysis", "gold", "search_analysis")

    descriptive_path = os.path.join(gold_base, "descriptive")
    diagnostic_path = os.path.join(gold_base, "diagnostic")
    predictive_path = os.path.join(gold_base, "predictive")
    prescriptive_path = os.path.join(gold_base, "prescriptive")
    query_perf_path = os.path.join(gold_base, "query_performance")
    channel_perf_path = os.path.join(gold_base, "channel_leaderboard")
    publish_hour_path = os.path.join(gold_base, "publish_hour_effect")
    video_scoring_path = os.path.join(gold_base, "video_scoring")

    os.makedirs(gold_base, exist_ok=True)

    df = spark.read.format("delta").load(silver_path).cache()

    # 1) DESCRIPTIVE ANALYSIS
    descriptive_df = (
        df.groupBy("category")
        .agg(
            count("video_id").alias("video_count"),
            avg("view_count").alias("avg_views"),
            avg("like_count").alias("avg_likes"),
            avg("comment_count").alias("avg_comments"),
            avg("engagement").alias("avg_engagement"),
        )
        .orderBy(desc("avg_views"))
    )
    descriptive_df.write.format("delta").mode("overwrite").save(descriptive_path)

    # 2) DIAGNOSTIC ANALYSIS
    views_likes_corr = df.stat.corr("view_count", "like_count")
    views_comments_corr = df.stat.corr("view_count", "comment_count")
    engagement_duration_corr = df.stat.corr("engagement", "duration_sec")

    diagnostic_rows = [
        ("view_count_like_count", float(views_likes_corr) if views_likes_corr is not None else 0.0),
        ("view_count_comment_count", float(views_comments_corr) if views_comments_corr is not None else 0.0),
        ("engagement_duration_sec", float(engagement_duration_corr) if engagement_duration_corr is not None else 0.0),
    ]
    diagnostic_df = spark.createDataFrame(diagnostic_rows, ["metric_pair", "correlation"])
    diagnostic_df.write.format("delta").mode("overwrite").save(diagnostic_path)

    # 3) PREDICTIVE ANALYSIS
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
        inputCols=["like_count", "comment_count", "duration_sec", "video_age_days", "title_length", "category_index"],
        outputCol="features",
    )
    ml_df = assembler.transform(indexed_df).select("features", col("view_count").alias("label"))

    train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
    lr = LinearRegression(featuresCol="features", labelCol="label")
    model = lr.fit(train_df)
    pred_df = (
        model.transform(test_df)
        .select(
            col("label").alias("actual_view_count"),
            col("prediction").alias("predicted_view_count"),
        )
    )
    pred_df.write.format("delta").mode("overwrite").save(predictive_path)

    # 4) PRESCRIPTIVE ANALYSIS
    prescriptive_df = (
        df.withColumn("publish_hour", hour(col("published_at")))
        .groupBy("category", "publish_hour")
        .agg(
            avg("view_count").alias("avg_views"),
            avg("engagement").alias("avg_engagement"),
        )
        .orderBy(desc("avg_views"))
    )
    best_slot_df = (
        prescriptive_df.groupBy("category")
        .agg(
            {"avg_views": "max", "avg_engagement": "max"}
        )
        .withColumnRenamed("max(avg_views)", "best_avg_views")
        .withColumnRenamed("max(avg_engagement)", "best_avg_engagement")
    )
    best_slot_df.write.format("delta").mode("overwrite").save(prescriptive_path)

    # 5) QUERY PERFORMANCE
    query_perf_df = (
        df.filter(col("search_query").isNotNull())
        .groupBy("search_query")
        .agg(
            count("video_id").alias("video_count"),
            avg("view_count").alias("avg_views"),
            avg("engagement").alias("avg_engagement"),
            avg("like_count").alias("avg_likes"),
        )
        .orderBy(desc("avg_views"))
    )
    query_perf_df.write.format("delta").mode("overwrite").save(query_perf_path)

    # 6) CHANNEL LEADERBOARD
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

    # 7) PUBLISH-HOUR EFFECT
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

    # 8) VIDEO SCORING
    video_scoring_df = (
        df.select(
            "video_id",
            "title",
            "channel_title",
            "category",
            "search_query",
            "view_count",
            "engagement",
            "video_age_days",
            "duration_sec",
            "published_at",
        )
        .withColumn(
            "score",
            (log1p(col("view_count")) * 0.55)
            + ((col("engagement") * 100.0) * 0.35)
            - (col("video_age_days") * 0.10),
        )
        .orderBy(desc("score"))
    )
    video_scoring_df.write.format("delta").mode("overwrite").save(video_scoring_path)

    print("Search analysis complete. Delta tables written:")
    print(f"- {descriptive_path}")
    print(f"- {diagnostic_path}")
    print(f"- {predictive_path}")
    print(f"- {prescriptive_path}")
    print(f"- {query_perf_path}")
    print(f"- {channel_perf_path}")
    print(f"- {publish_hour_path}")
    print(f"- {video_scoring_path}")

    spark.stop()


if __name__ == "__main__":
    main()
