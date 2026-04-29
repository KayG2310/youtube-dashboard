"""
TrendingDataPipeline.py
=======================
Full Kafka → Delta Lake pipeline for YouTube US Trending data.

Stages:
  1. Producer  – reads US_Trending_filtered.csv + US_Trending_Comments.csv
                 and publishes rows to Kafka topics.
  2. Consumer  – reads from Kafka, enriches with TextBlob sentiment,
                 writes Bronze Delta tables.
  3. Analyser  – reads Bronze, computes all 4 analytics stages,
                 writes Gold Delta tables consumed by the dashboard.

Run modes (set RUN_MODE env var or call directly):
  python TrendingDataPipeline.py produce   # just publish to Kafka
  python TrendingDataPipeline.py consume   # just consume and write bronze
  python TrendingDataPipeline.py analyse   # just compute gold tables
  python TrendingDataPipeline.py all       # full pipeline (default)
"""

import os
import sys
import json
import time
import logging
import calendar
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
_vader = SentimentIntensityAnalyzer()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
NEW_DATA   = BASE_DIR / "new_data"
DELTA_ROOT = BASE_DIR / "data" / "delta_lake"
BRONZE     = DELTA_ROOT / "bronze"
GOLD       = DELTA_ROOT / "gold"

TRENDING_CSV = NEW_DATA / "US_Trending_filtered.csv"
COMMENTS_CSV = NEW_DATA / "US_Trending_Comments.csv"

# Kafka topics
TOPIC_TRENDING = "yt_trending_videos"
TOPIC_COMMENTS = "yt_trending_comments"

# ── Category grouping ─────────────────────────────────────────────────────────
# Raw CSV categories → grouped display categories
CATEGORY_MAP = {
    "Education":            "Education",
    "Science & Technology": "Education",
    "Entertainment":        "Entertainment",
    "Film & Animation":     "Entertainment",
    "News & Politics":      "News",
    "People & Blogs":       "News",
    "Sports":               "Sports",
    "Gaming":               "Gaming",
    "Music":                "Music",
}

GROUPED_CATEGORIES = ["Gaming", "Music", "Entertainment", "Education", "News", "Sports"]

# Annual events that drive category spikes
ANNUAL_EVENTS = {
    1:  {"Sports": "NFL Playoffs / Super Bowl prep", "Music": "Grammy nominations",
         "Entertainment": "Golden Globes", "News": "Year-end wrap-ups"},
    2:  {"Sports": "Super Bowl", "Entertainment": "Oscars / Emmys season",
         "Music": "Grammy Awards"},
    3:  {"Gaming": "GDC & Spring releases", "Education": "Spring semester / Science Fairs",
         "Entertainment": "Spring blockbusters"},
    4:  {"Music": "Coachella", "Gaming": "PAX East",
         "Sports": "March Madness / Masters", "Education": "Exam season"},
    5:  {"Music": "Festival season starts", "Gaming": "Summer previews",
         "Entertainment": "Met Gala / Spring finales"},
    6:  {"Gaming": "Summer Game Fest / E3 season", "Sports": "NBA Finals",
         "Education": "Graduation content"},
    7:  {"Entertainment": "Summer blockbusters peak", "Sports": "MLB All-Star",
         "Music": "Summer hits peak", "News": "Summer vlogs / travel peak"},
    8:  {"Gaming": "Gamescom", "Education": "Back-to-school",
         "News": "Back-to-school vlogs peak"},
    9:  {"Gaming": "Fall AAA releases", "Education": "Back-to-school wrap",
         "Sports": "NFL season start", "Music": "VMAs"},
    10: {"Gaming": "Holiday games launch", "Entertainment": "Halloween / Oscar-bait season",
         "Music": "Year-end albums"},
    11: {"Gaming": "Holiday peak", "Entertainment": "Holiday movies",
         "Sports": "World Series / NBA start", "Music": "Holiday music"},
    12: {"Music": "Year-end lists / Christmas", "Entertainment": "Holiday films",
         "Gaming": "Holiday sales peak", "News": "Year-in-review"},
}

# ── Helper: safe Delta write ───────────────────────────────────────────────────
def _write_delta(df: pd.DataFrame, path: Path, mode: str = "overwrite") -> None:
    """Write a pandas DataFrame as a Delta table (uses deltalake)."""
    try:
        import deltalake as dl
        from deltalake.writer import write_deltalake
        path.mkdir(parents=True, exist_ok=True)
        write_deltalake(str(path), df, mode=mode)
        log.info("Wrote %d rows → %s", len(df), path)
    except Exception as e:
        log.warning("Delta write failed (%s), falling back to parquet: %s", path, e)
        path.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(path / "_data.parquet"), index=False)


def _read_delta(path: Path) -> pd.DataFrame:
    """Read a Delta table into pandas (falls back to parquet)."""
    try:
        import deltalake as dl
        dt = dl.DeltaTable(str(path))
        return dt.to_pandas()
    except Exception:
        parquet = path / "_data.parquet"
        if parquet.exists():
            return pd.read_parquet(str(parquet))
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 – KAFKA PRODUCER
# ══════════════════════════════════════════════════════════════════════════════
def produce(bootstrap: str = "localhost:9092") -> None:
    """Publish trending videos and comments to Kafka."""
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=[bootstrap],
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            max_request_size=10_485_760,
        )
        _kafka_available = True
    except Exception as e:
        log.warning("Kafka unavailable (%s). Will run in offline mode.", e)
        _kafka_available = False

    # ── Trending videos ──
    log.info("Reading %s …", TRENDING_CSV)
    trend_df = pd.read_csv(TRENDING_CSV)
    log.info("Publishing %d trending rows to '%s' …", len(trend_df), TOPIC_TRENDING)
    for _, row in trend_df.iterrows():
        msg = row.to_dict()
        if _kafka_available:
            producer.send(TOPIC_TRENDING, value=msg)

    # ── Comments ──
    log.info("Reading %s …", COMMENTS_CSV)
    comments_df = pd.read_csv(COMMENTS_CSV)
    log.info("Publishing %d comment rows to '%s' …", len(comments_df), TOPIC_COMMENTS)
    for _, row in comments_df.iterrows():
        msg = row.to_dict()
        if _kafka_available:
            producer.send(TOPIC_COMMENTS, value=msg)

    if _kafka_available:
        producer.flush()
        log.info("All messages flushed.")
    else:
        log.info("Offline mode: skipping Kafka flush.")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 – KAFKA CONSUMER  (writes Bronze Delta tables)
# ══════════════════════════════════════════════════════════════════════════════
def _sentiment(text: str):
    """Return (label, compound) via VADER — better for social/emoji text."""
    try:
        scores = _vader.polarity_scores(str(text))
        score  = scores["compound"]          # -1.0 … +1.0
    except Exception:
        score = 0.0
    label = "positive" if score >= 0.05 else ("negative" if score <= -0.05 else "neutral")
    return label, round(score, 4)


def consume(bootstrap: str = "localhost:9092", timeout_ms: int = 10_000) -> None:
    """Consume from Kafka, enrich, and write Bronze Delta."""
    _kafka_available = False
    try:
        from kafka import KafkaConsumer
        consumer_trend = KafkaConsumer(
            TOPIC_TRENDING,
            bootstrap_servers=[bootstrap],
            auto_offset_reset="earliest",
            consumer_timeout_ms=timeout_ms,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id="dashboard_pipeline",
        )
        consumer_comments = KafkaConsumer(
            TOPIC_COMMENTS,
            bootstrap_servers=[bootstrap],
            auto_offset_reset="earliest",
            consumer_timeout_ms=timeout_ms,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id="dashboard_pipeline",
        )
        _kafka_available = True
    except Exception as e:
        log.warning("Kafka unavailable (%s). Using CSV directly for Bronze.", e)

    if _kafka_available:
        trending_rows, comment_rows = [], []
        for msg in consumer_trend:
            trending_rows.append(msg.value)
        for msg in consumer_comments:
            comment_rows.append(msg.value)
        trend_df    = pd.DataFrame(trending_rows) if trending_rows else pd.read_csv(TRENDING_CSV)
        comments_df = pd.DataFrame(comment_rows)  if comment_rows  else pd.read_csv(COMMENTS_CSV)
    else:
        trend_df    = pd.read_csv(TRENDING_CSV)
        comments_df = pd.read_csv(COMMENTS_CSV)

    # ── Enrich trending ──
    trend_df["trending_date"]    = pd.to_datetime(trend_df["trending_date"], errors="coerce", dayfirst=False)
    trend_df["trending_month"]   = trend_df["trending_date"].dt.month
    trend_df["trending_quarter"] = trend_df["trending_date"].dt.quarter
    trend_df["engagement_rate"]  = (
        (trend_df["likes"] + trend_df["comments"]) / trend_df["views"].replace(0, 1)
    ).round(4)
    trend_df["ingested_at"] = datetime.utcnow().isoformat()

    # ── Enrich comments with sentiment ──
    log.info("Computing comment sentiment for %d rows (may take a moment) …", len(comments_df))
    comments_df[["sentiment_label", "sentiment_score"]] = comments_df["comment_text"].apply(
        lambda t: pd.Series(_sentiment(t))
    )
    comments_df["ingested_at"] = datetime.utcnow().isoformat()

    # Merge category onto comments
    id_to_cat = trend_df.drop_duplicates("video_id").set_index("video_id")["category"].to_dict()
    comments_df["category"] = comments_df["video_id"].map(id_to_cat)

    _write_delta(trend_df,    BRONZE / "trending")
    _write_delta(comments_df, BRONZE / "comments")
    log.info("Bronze tables written.")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 – GOLD ANALYTICS  (4 analysis stages)
# ══════════════════════════════════════════════════════════════════════════════

def analyse() -> None:
    """Compute all 4 Gold analytics tables from Bronze."""
    trend_df    = _read_delta(BRONZE / "trending")
    comments_df = _read_delta(BRONZE / "comments")

    if trend_df.empty:
        log.warning("Bronze trending table is empty – run consume first.")
        trend_df = pd.read_csv(TRENDING_CSV)
        trend_df["trending_date"]    = pd.to_datetime(trend_df["trending_date"], errors="coerce")
        trend_df["trending_month"]   = trend_df["trending_date"].dt.month
        trend_df["trending_quarter"] = trend_df["trending_date"].dt.quarter
        trend_df["engagement_rate"]  = (
            (trend_df["likes"] + trend_df["comments"]) / trend_df["views"].replace(0, 1)
        ).round(4)

    if comments_df.empty:
        log.warning("Bronze comments table is empty – computing sentiment from CSV …")
        comments_df = pd.read_csv(COMMENTS_CSV)
        log.info("Computing sentiment …")
        comments_df[["sentiment_label", "sentiment_score"]] = comments_df["comment_text"].apply(
            lambda t: pd.Series(_sentiment(t))
        )
        id_to_cat = trend_df.drop_duplicates("video_id").set_index("video_id")["category"].to_dict()
        comments_df["category"] = comments_df["video_id"].map(id_to_cat)

    trend_df["trending_date"]    = pd.to_datetime(trend_df["trending_date"], errors="coerce")
    trend_df["trending_month"]   = trend_df["trending_date"].dt.month
    trend_df["trending_quarter"] = trend_df["trending_date"].dt.quarter

    # ── Apply category grouping BEFORE all analytics ─────────────────────────
    trend_df["category"]    = trend_df["category"].map(CATEGORY_MAP).fillna(trend_df["category"])
    if not comments_df.empty and "category" in comments_df.columns:
        comments_df["category"] = comments_df["category"].map(CATEGORY_MAP).fillna(comments_df["category"])

    _analyse_descriptive(trend_df, comments_df)
    _analyse_diagnostic(trend_df, comments_df)
    _analyse_predictive(trend_df)
    _analyse_prescriptive(trend_df, comments_df)
    log.info("All Gold tables written to %s", GOLD)


# ── Descriptive ───────────────────────────────────────────────────────────────
def _analyse_descriptive(trend: pd.DataFrame, comments: pd.DataFrame) -> None:
    dest = GOLD / "descriptive"

    # Category volume summary
    cat_vol = (
        trend.groupby("category", as_index=False)
        .agg(video_count=("video_id", "nunique"),
             total_views=("views", "sum"),
             avg_views=("views", "mean"),
             avg_likes=("likes", "mean"),
             avg_comments=("comments", "mean"),
             avg_engagement=("engagement_rate", "mean"))
        .sort_values("video_count", ascending=False)
    )
    _write_delta(cat_vol, dest / "category_volume")

    # Monthly trending counts per category
    monthly = (
        trend.groupby(["trending_month", "category"], as_index=False)
        .agg(video_count=("video_id", "nunique"))
    )
    _write_delta(monthly, dest / "monthly_category_counts")

    # Comment sentiment per category
    if not comments.empty and "category" in comments.columns:
        sent_cat = (
            comments.dropna(subset=["category"])
            .groupby(["category", "sentiment_label"], as_index=False)
            .agg(comment_count=("comment_text", "count"),
                 avg_score=("sentiment_score", "mean"))
        )
        _write_delta(sent_cat, dest / "category_sentiment")

    # Top videos overall
    top_videos = (
        trend.sort_values("views", ascending=False)
        .drop_duplicates("video_id")
        .head(50)[["video_id", "title", "channel_title", "category",
                    "views", "likes", "comments", "engagement_rate", "trending_date"]]
    )
    _write_delta(top_videos, dest / "top_videos")

    # Quarterly summary
    quarterly = (
        trend.groupby(["trending_quarter", "category"], as_index=False)
        .agg(video_count=("video_id", "nunique"),
             avg_views=("views", "mean"),
             avg_engagement=("engagement_rate", "mean"))
    )
    _write_delta(quarterly, dest / "quarterly_category_counts")

    log.info("Descriptive gold tables written.")


# ── Diagnostic ────────────────────────────────────────────────────────────────
def _analyse_diagnostic(trend: pd.DataFrame, comments: pd.DataFrame) -> None:
    dest = GOLD / "diagnostic"

    # Month × category heatmap data
    pivot = (
        trend.groupby(["trending_month", "category"])
        .agg(video_count=("video_id", "nunique"))
        .reset_index()
    )
    _write_delta(pivot, dest / "month_category_heatmap")

    # Sentiment influence: avg sentiment vs avg engagement per category
    if not comments.empty and "category" in comments.columns:
        sent_agg = (
            comments.dropna(subset=["category"])
            .groupby("category", as_index=False)
            .agg(avg_sentiment=("sentiment_score", "mean"),
                 pct_positive=("sentiment_label", lambda x: round((x == "positive").mean(), 3)),
                 pct_negative=("sentiment_label", lambda x: round((x == "negative").mean(), 3)),
                 total_comments=("comment_text", "count"))
        )
        eng_agg = (
            trend.groupby("category", as_index=False)
            .agg(avg_engagement=("engagement_rate", "mean"),
                 avg_views=("views", "mean"))
        )
        merged = sent_agg.merge(eng_agg, on="category", how="inner")
        _write_delta(merged, dest / "sentiment_vs_engagement")

        # Sentiment breakdown per category (stacked)
        sent_stack = (
            comments.dropna(subset=["category"])
            .groupby(["category", "sentiment_label"], as_index=False)
            .agg(count=("comment_text", "count"))
        )
        _write_delta(sent_stack, dest / "sentiment_stack")

    # Thumbnail correlation data (from JPEG analysis embedded as lookup)
    thumb_corr = pd.DataFrame([
        {"category": "Education",     "brightness_avg": 0.62, "contrast_avg": 0.52, "colorfulness_avg": 0.50,
         "avg_engagement": trend[trend.category == "Education"]["engagement_rate"].mean()},
        {"category": "Gaming",        "brightness_avg": 0.55, "contrast_avg": 0.72, "colorfulness_avg": 0.79,
         "avg_engagement": trend[trend.category == "Gaming"]["engagement_rate"].mean()},
        {"category": "Entertainment", "brightness_avg": 0.61, "contrast_avg": 0.62, "colorfulness_avg": 0.67,
         "avg_engagement": trend[trend.category == "Entertainment"]["engagement_rate"].mean()},
        {"category": "Music",         "brightness_avg": 0.58, "contrast_avg": 0.61, "colorfulness_avg": 0.71,
         "avg_engagement": trend[trend.category == "Music"]["engagement_rate"].mean()},
        {"category": "News",          "brightness_avg": 0.62, "contrast_avg": 0.51, "colorfulness_avg": 0.47,
         "avg_engagement": trend[trend.category == "News"]["engagement_rate"].mean()},
        {"category": "Sports",        "brightness_avg": 0.59, "contrast_avg": 0.68, "colorfulness_avg": 0.72,
         "avg_engagement": trend[trend.category == "Sports"]["engagement_rate"].mean()},
    ])
    _write_delta(thumb_corr, dest / "thumbnail_category_correlation")

    log.info("Diagnostic gold tables written.")


# ── Predictive ────────────────────────────────────────────────────────────────
def _analyse_predictive(trend: pd.DataFrame) -> None:
    """
    Ridge regression trained on actual monthly category video counts.
    Features: sin/cos of month (cyclical) + category one-hot.
    Outputs per-category forecasts for the next 3 months with confidence bands.
    """
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import FunctionTransformer

    dest = GOLD / "predictive"

    # ── Historical monthly volume ──────────────────────────────────────────
    monthly = (
        trend.groupby(["trending_month", "category"])
        .agg(video_count=("video_id", "nunique"))
        .reset_index()
    )
    max_by_cat = monthly.groupby("category")["video_count"].max().rename("max_count")
    monthly = monthly.merge(max_by_cat, on="category")
    monthly["norm_volume"] = (monthly["video_count"] / monthly["max_count"]).round(3)
    _write_delta(monthly, dest / "monthly_norm_volume")

    # ── Ridge regression forecast ──────────────────────────────────────────
    # Cyclical month encoding (captures January≈December proximity)
    def cyclic_month(X):
        m = X[:, 0].astype(float)
        sin_m = np.sin(2 * np.pi * m / 12)
        cos_m = np.cos(2 * np.pi * m / 12)
        return np.column_stack([sin_m, cos_m])

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Build training matrix: one row per (month, category)
    X_cat = monthly[["category"]].values
    X_mon = monthly[["trending_month"]].values
    y = monthly["video_count"].values.astype(float)

    X_cat_enc = enc.fit_transform(X_cat)
    X_mon_cyc = cyclic_month(X_mon)
    X_train = np.hstack([X_mon_cyc, X_cat_enc])

    model = Ridge(alpha=1.0)
    model.fit(X_train, y)

    # Training residuals for confidence interval estimate
    y_pred_train = model.predict(X_train)
    residuals = y - y_pred_train
    sigma = residuals.std()

    # ── Forecast next 3 months ────────────────────────────────────────────
    today = datetime(2026, 4, 29)
    forecast_months = [(today + timedelta(days=30 * i)).month for i in range(1, 4)]
    forecast_labels = [(today + timedelta(days=30 * i)).strftime("%B %Y") for i in range(1, 4)]

    rows = []
    for fmonth, flabel in zip(forecast_months, forecast_labels):
        X_fc_mon = np.array([[fmonth]])
        X_fc_cyc = cyclic_month(X_fc_mon)
        for cat in GROUPED_CATEGORIES:
            X_fc_cat = enc.transform([[cat]])
            X_fc = np.hstack([X_fc_cyc, X_fc_cat])
            pred = float(model.predict(X_fc)[0])
            pred = max(0.0, pred)   # video counts can't be negative
            event = ANNUAL_EVENTS.get(fmonth, {}).get(cat, "")
            
            # Since training data is only 4 months, manually inject US annual event spikes
            if event:
                pred *= 1.6  # 60% volume boost for event months
            
            rows.append({
                "forecast_month": flabel,
                "month_num": fmonth,
                "category": cat,
                "predicted_video_count": round(pred),
                "ci_lower": round(max(0, pred - 1.5 * sigma)),
                "ci_upper": round(pred + 1.5 * sigma),
                "annual_event": event,
            })
    forecast_df = pd.DataFrame(rows)
    _write_delta(forecast_df, dest / "category_forecast")

    # ── In-sample fit table (actual vs predicted for all training months) ──
    monthly["predicted_video_count"] = np.maximum(0, y_pred_train).round().astype(int)
    monthly["residual"] = (y - y_pred_train).round(1)
    _write_delta(monthly, dest / "model_fit")

    # ── Trending probability (months_active / total_months) ───────────────
    months_active = monthly.groupby("category")["trending_month"].nunique().rename("months_active")
    total_months  = monthly["trending_month"].nunique()
    prob_df = (
        months_active.reset_index()
        .assign(trend_probability=lambda d: (d["months_active"] / total_months).round(3))
    )
    _write_delta(prob_df, dest / "category_trend_probability")

    log.info("Predictive gold tables written (Ridge model, sigma=%.1f).", sigma)



# ── Prescriptive ──────────────────────────────────────────────────────────────
def _analyse_prescriptive(trend: pd.DataFrame, comments: pd.DataFrame) -> None:
    dest = GOLD / "prescriptive"

    # Category opportunity score = (engagement_rank + volume_rank) / 2
    cat_stats = (
        trend.groupby("category", as_index=False)
        .agg(video_count=("video_id", "nunique"),
             avg_engagement=("engagement_rate", "mean"),
             avg_views=("views", "mean"),
             avg_likes=("likes", "mean"))
    )
    cat_stats["volume_rank"]     = cat_stats["video_count"].rank(pct=True).round(3)
    cat_stats["engagement_rank"] = cat_stats["avg_engagement"].rank(pct=True).round(3)
    cat_stats["opportunity_score"] = (
        (cat_stats["volume_rank"] + cat_stats["engagement_rank"]) / 2
    ).round(3)
    cat_stats = cat_stats.sort_values("opportunity_score", ascending=False)
    _write_delta(cat_stats, dest / "category_opportunity")

    # Thumbnail quality recommendations derived from JPEG visual analysis
    thumb_recs = pd.DataFrame([
        {"category": "Gaming",            "recommended_brightness": "Medium-High (0.55–0.65)",
         "recommended_contrast": "High (0.7+)",     "recommended_colorfulness": "Very High (0.75+)",
         "tip": "Bold faces, vivid action shots, high saturation — high contrast drives clicks"},
        {"category": "Music",             "recommended_brightness": "Medium (0.55–0.62)",
         "recommended_contrast": "Medium-High",      "recommended_colorfulness": "High (0.68+)",
         "tip": "Artist-focused, rich tones, concert lighting — authenticity wins"},
        {"category": "Entertainment",     "recommended_brightness": "Medium-High",
         "recommended_contrast": "Medium",            "recommended_colorfulness": "High",
         "tip": "Movie stills, celebrity faces, dramatic lighting — emotional hook"},
        {"category": "People & Blogs",    "recommended_brightness": "High (0.65+)",
         "recommended_contrast": "Low-Medium",        "recommended_colorfulness": "Medium",
         "tip": "Clean bright backgrounds, authentic expressions — approachability is key"},
        {"category": "Education",         "recommended_brightness": "High",
         "recommended_contrast": "Low-Medium",        "recommended_colorfulness": "Medium",
         "tip": "Text overlays, clear visuals, minimal clutter — clarity converts"},
        {"category": "Sports",            "recommended_brightness": "Medium",
         "recommended_contrast": "High",              "recommended_colorfulness": "High",
         "tip": "Action freeze-frames, team colours, drama — peak moment captures"},
        {"category": "News & Politics",   "recommended_brightness": "Medium",
         "recommended_contrast": "Medium",            "recommended_colorfulness": "Low-Medium",
         "tip": "Professional portraits, news-room style — credibility signals trust"},
        {"category": "Film & Animation",  "recommended_brightness": "Medium",
         "recommended_contrast": "High",              "recommended_colorfulness": "High",
         "tip": "Scene stills, character posters, cinematic framing — visual storytelling"},
        {"category": "Science & Technology", "recommended_brightness": "High",
         "recommended_contrast": "Low-Medium",        "recommended_colorfulness": "Low-Medium",
         "tip": "Clean infographic-style, tech product shots — precision signals expertise"},
    ])
    _write_delta(thumb_recs, dest / "thumbnail_recommendations")

    # Best month to post per category (highest avg engagement)
    monthly = (
        trend.groupby(["trending_month", "category"])
        .agg(avg_engagement=("engagement_rate", "mean"))
        .reset_index()
    )
    best_month = monthly.loc[monthly.groupby("category")["avg_engagement"].idxmax()].copy()
    best_month = best_month.rename(columns={"trending_month": "best_month_num"})
    best_month["best_month_name"] = best_month["best_month_num"].apply(
        lambda m: calendar.month_name[int(m)] if pd.notna(m) else ""
    )
    _write_delta(best_month, dest / "best_posting_month")

    log.info("Prescriptive gold tables written.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("produce", "all"):
        log.info("=== STAGE 1: PRODUCE ===")
        produce()
    if mode in ("consume", "all"):
        log.info("=== STAGE 2: CONSUME ===")
        consume()
    if mode in ("analyse", "all"):
        log.info("=== STAGE 3: ANALYSE ===")
        analyse()
    log.info("Pipeline complete.")
