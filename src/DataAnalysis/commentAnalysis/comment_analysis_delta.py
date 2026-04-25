import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, desc, when, stddev, corr

def create_spark():
    return (
        SparkSession.builder
        .appName("CommentGoldAnalysis")
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
    comments_silver_path = os.path.join(base_dir, "data", "processed", "silver", "comments")
    search_silver_path = os.path.join(base_dir, "data", "processed", "silver", "search")
    
    gold_base = os.path.join(base_dir, "data", "analysis", "gold", "comment_analysis")
    
    video_sentiment_path = os.path.join(gold_base, "video_sentiment_summary")
    lang_sentiment_path = os.path.join(gold_base, "language_sentiment_summary")
    sentiment_impact_path = os.path.join(gold_base, "sentiment_impact_analysis")
    top_negative_videos_path = os.path.join(gold_base, "top_negative_videos")
    
    os.makedirs(gold_base, exist_ok=True)

    if not os.path.exists(comments_silver_path):
        print(f"Comments Silver table not found at {comments_silver_path}. Skipping analysis.")
        spark.stop()
        return

    comments_df = spark.read.format("delta").load(comments_silver_path).cache()

    # 1) VIDEO SENTIMENT SUMMARY
    # Aggregating sentiment metrics per video
    raw_video_sentiment = (
        comments_df.groupBy("video_id")
        .agg(
            count("comment_id").alias("total_comments"),
            avg("sentiment_score").alias("avg_sentiment"),
            stddev("sentiment_score").alias("sentiment_volatility"),
            avg("like_count").alias("avg_comment_likes"),
            count(when(col("sentiment_label") == "positive", True)).alias("positive_count"),
            count(when(col("sentiment_label") == "negative", True)).alias("negative_count"),
            count(when(col("sentiment_label") == "neutral", True)).alias("neutral_count")
        )
        .withColumn("positivity_ratio", col("positive_count") / col("total_comments"))
    )

    # Join with Search Silver to get titles for the summary
    if os.path.exists(search_silver_path):
        search_df = spark.read.format("delta").load(search_silver_path)
        video_sentiment_df = (
            raw_video_sentiment.join(search_df.select("video_id", "title", "channel_title"), "video_id", "left")
            .orderBy(desc("avg_sentiment"))
        )
    else:
        video_sentiment_df = raw_video_sentiment.orderBy(desc("avg_sentiment"))

    video_sentiment_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(video_sentiment_path)

    # 2) LANGUAGE SENTIMENT SUMMARY
    lang_sentiment_df = (
        comments_df.groupBy("language")
        .agg(
            count("comment_id").alias("comment_count"),
            avg("sentiment_score").alias("avg_sentiment"),
            avg("like_count").alias("avg_likes")
        )
        .orderBy(desc("comment_count"))
    )
    lang_sentiment_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(lang_sentiment_path)

    # 3) SENTIMENT IMPACT ANALYSIS (Joining with Search Silver for Video Context)
    if os.path.exists(search_silver_path):
        search_df = spark.read.format("delta").load(search_silver_path)
        
        impact_df = (
            video_sentiment_df.join(search_df.select("video_id", "view_count", "category"), "video_id")
            .select(
                "video_id", "title", "channel_title", "category",
                "avg_sentiment", "total_comments", "view_count"
            )
            .orderBy(desc("view_count"))
        )
        impact_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(sentiment_impact_path)
        
        # 4) TOP NEGATIVE VIDEOS (For moderation/intervention)
        negative_videos = (
            impact_df.filter(col("total_comments") > 5) # Filter out videos with too few comments
            .orderBy("avg_sentiment")
            .limit(20)
        )
        negative_videos.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(top_negative_videos_path)
    else:
        print("Search Silver table not found. Skipping Impact Analysis.")

    print("Comment Gold analysis complete. Delta tables written:")
    print(f"- {video_sentiment_path}")
    print(f"- {lang_sentiment_path}")
    if os.path.exists(search_silver_path):
        print(f"- {sentiment_impact_path}")
        print(f"- {top_negative_videos_path}")

    spark.stop()

if __name__ == "__main__":
    main()
