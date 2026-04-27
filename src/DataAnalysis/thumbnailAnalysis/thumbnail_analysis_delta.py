import os
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, desc, hour, log1p, max as spark_max


def create_spark():
    return (
        SparkSession.builder
        .appName("ThumbnailGoldAnalysis")
        .master("local[*]")
        .config(
            "spark.jars.packages",
            "io.delta:delta-spark_2.12:3.0.0",
        )
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


def _safe_corr(df, left, right):
    value = df.stat.corr(left, right)
    return float(value) if value is not None else 0.0


def main():
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    silver_path = os.path.join(base_dir, "data", "processed", "silver", "thumbnail")
    search_silver_path = os.path.join(base_dir, "data", "processed", "silver", "search")
    gold_base = os.path.join(base_dir, "data", "analysis", "gold", "thumbnail_analysis")

    descriptive_path = os.path.join(gold_base, "descriptive")
    diagnostic_path = os.path.join(gold_base, "diagnostic")
    predictive_path = os.path.join(gold_base, "predictive")
    prescriptive_path = os.path.join(gold_base, "prescriptive")
    video_scoring_path = os.path.join(gold_base, "video_scoring")

    os.makedirs(gold_base, exist_ok=True)

    if not os.path.exists(silver_path):
        print(f"Thumbnail Silver table not found at {silver_path}. Skipping thumbnail gold analysis.")
        spark.stop()
        return

    thumbnail_df = spark.read.format("delta").load(silver_path).cache()

    search_joined = False
    if os.path.exists(search_silver_path):
        search_df = spark.read.format("delta").load(search_silver_path).select(
            "video_id",
            "view_count",
            "like_count",
            "engagement",
            "published_at",
            "video_age_days",
        )
        joined_df = thumbnail_df.join(search_df, on="video_id", how="left")
        search_joined = True
    else:
        joined_df = thumbnail_df

    descriptive_df = (
        thumbnail_df.groupBy("dominant_color")
        .agg(
            count("video_id").alias("thumbnail_count"),
            avg("brightness").alias("avg_brightness"),
            avg("contrast").alias("avg_contrast"),
            avg("colorfulness").alias("avg_colorfulness"),
            avg("sharpness").alias("avg_sharpness"),
            avg("thumbnail_quality_score").alias("avg_quality_score"),
        )
        .orderBy(desc("thumbnail_count"))
    )
    descriptive_df.write.format("delta").mode("overwrite").save(descriptive_path)

    diagnostic_rows = [
        ("quality_brightness", _safe_corr(thumbnail_df, "thumbnail_quality_score", "brightness")),
        ("quality_contrast", _safe_corr(thumbnail_df, "thumbnail_quality_score", "contrast")),
        ("quality_colorfulness", _safe_corr(thumbnail_df, "thumbnail_quality_score", "colorfulness")),
        ("quality_sharpness", _safe_corr(thumbnail_df, "thumbnail_quality_score", "sharpness")),
    ]

    if search_joined:
        diagnostic_rows.extend([
            ("quality_view_count", _safe_corr(joined_df, "thumbnail_quality_score", "view_count")),
            ("quality_like_count", _safe_corr(joined_df, "thumbnail_quality_score", "like_count")),
            ("quality_engagement", _safe_corr(joined_df, "thumbnail_quality_score", "engagement")),
        ])

    diagnostic_df = spark.createDataFrame(diagnostic_rows, ["metric_pair", "correlation"])
    diagnostic_df.write.format("delta").mode("overwrite").save(diagnostic_path)

    if search_joined and joined_df.filter(col("view_count").isNotNull()).count() > 0:
        model_input = (
            joined_df.select(
                "brightness",
                "contrast",
                "colorfulness",
                "sharpness",
                "thumbnail_quality_score",
                col("view_count").alias("label"),
            )
            .dropna()
        )

        assembler = VectorAssembler(
            inputCols=["brightness", "contrast", "colorfulness", "sharpness", "thumbnail_quality_score"],
            outputCol="features",
        )
        ml_df = assembler.transform(model_input).select("features", "label")

        train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
        lr = LinearRegression(featuresCol="features", labelCol="label")
        lr_model = lr.fit(train_df)
        pred_df = lr_model.transform(test_df).select(col("label").alias("actual_view_count"), col("prediction").alias("predicted_view_count"))
    else:
        pred_df = spark.createDataFrame([], "actual_view_count double, predicted_view_count double")

    pred_df.write.format("delta").mode("overwrite").save(predictive_path)

    if search_joined and joined_df.filter(col("published_at").isNotNull()).count() > 0:
        prescriptive_df = (
            joined_df.withColumn("publish_hour", hour(col("published_at")))
            .groupBy("publish_hour")
            .agg(
                avg("view_count").alias("avg_views"),
                avg("engagement").alias("avg_engagement"),
            )
            .orderBy(desc("avg_views"))
        )
    else:
        prescriptive_df = spark.createDataFrame([], "publish_hour int, avg_views double, avg_engagement double")

    prescriptive_df.write.format("delta").mode("overwrite").save(prescriptive_path)

    if search_joined:
        video_scoring_df = (
            joined_df.select(
                "video_id",
                "thumbnail_url",
                "dominant_color",
                "view_count",
                "like_count",
                "engagement",
                "thumbnail_quality_score",
                "published_at",
                "video_age_days",
            )
            .withColumn(
                "score",
                (log1p(col("view_count")) * 0.40)
                + (col("thumbnail_quality_score") * 0.35)
                + ((col("engagement") * 100.0) * 0.20)
                - (log1p(col("video_age_days") + 1) * 0.05)
            )
            .orderBy(desc("score"))
        )
    else:
        video_scoring_df = (
            thumbnail_df.select(
                "video_id",
                "thumbnail_url",
                "dominant_color",
                "thumbnail_quality_score",
                "brightness",
                "contrast",
                "colorfulness",
                "sharpness",
            )
            .withColumn(
                "score",
                (col("thumbnail_quality_score") * 0.75)
                + (log1p(col("brightness") + col("contrast") + col("colorfulness") + col("sharpness")) * 0.25)
            )
            .orderBy(desc("score"))
        )

    video_scoring_df.write.format("delta").mode("overwrite").save(video_scoring_path)

    print("Thumbnail Gold analysis complete. Delta tables written:")
    print(f"- {descriptive_path}")
    print(f"- {diagnostic_path}")
    print(f"- {predictive_path}")
    print(f"- {prescriptive_path}")
    print(f"- {video_scoring_path}")

    spark.stop()


if __name__ == "__main__":
    main()