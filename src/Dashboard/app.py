import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import deltalake
from pyspark.sql import SparkSession

# Set Page Config
st.set_page_config(page_title="YouTube Comment Analytics", layout="wide", page_icon="📊")

# --- DATA LOADING ---
@st.cache_data
def load_comments_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "processed", "comments_processed.csv")
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)


@st.cache_data
def load_search_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "processed", "search_processed.csv")
    if not os.path.exists(data_path):
        return None
    df = pd.read_csv(data_path)
    for col in ("published_at", "fetched_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


@st.cache_data
def load_thumbnail_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "final_analytics_report.json")
    if not os.path.exists(data_path):
        return None
    with open(data_path, "r") as f:
        return json.load(f)


@st.cache_data
def load_thumbnail_delta_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    delta_path = os.path.join(base_dir, "data", "processed", "silver", "thumbnail")
    if not os.path.exists(delta_path):
        return None
    dt = deltalake.DeltaTable(delta_path)
    df = dt.to_pandas()
    
    # Compute analytics
    analytics = {
        "avg_brightness": df["brightness"].mean(),
        "avg_contrast": df["contrast"].mean(),
        "avg_colorfulness": df["colorfulness"].mean(),
        "avg_sharpness": df["sharpness"].mean(),
        "avg_quality_score": df["thumbnail_quality_score"].mean(),
        "top_dominant_colors": df["dominant_color"].value_counts().head(10).to_dict(),
        "brightness_distribution": df["brightness"].describe().to_dict(),
        "total_thumbnails": len(df)
    }
    return {"thumbnail_analysis": analytics}


@st.cache_data
def load_search_delta_gold_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gold_base = os.path.join(base_dir, "data", "analysis", "gold", "search_analysis")
    paths = {
        "descriptive": os.path.join(gold_base, "descriptive"),
        "diagnostic": os.path.join(gold_base, "diagnostic"),
        "predictive": os.path.join(gold_base, "predictive"),
        "prescriptive": os.path.join(gold_base, "prescriptive"),
        "query_performance": os.path.join(gold_base, "query_performance"),
        "channel_leaderboard": os.path.join(gold_base, "channel_leaderboard"),
        "publish_hour_effect": os.path.join(gold_base, "publish_hour_effect"),
        "video_scoring": os.path.join(gold_base, "video_scoring"),
    }

    required = ["descriptive", "diagnostic", "predictive", "prescriptive"]
    if not all(os.path.exists(paths[k]) for k in required):
        return None

    spark = (
        SparkSession.builder
        .appName("SearchDeltaDashboardReader")
        .master("local[*]")
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        data = {
            "descriptive": spark.read.format("delta").load(paths["descriptive"]).toPandas(),
            "diagnostic": spark.read.format("delta").load(paths["diagnostic"]).toPandas(),
            "predictive": spark.read.format("delta").load(paths["predictive"]).toPandas(),
            "prescriptive": spark.read.format("delta").load(paths["prescriptive"]).toPandas(),
            "query_performance": spark.read.format("delta").load(paths["query_performance"]).toPandas() if os.path.exists(paths["query_performance"]) else pd.DataFrame(),
            "channel_leaderboard": spark.read.format("delta").load(paths["channel_leaderboard"]).toPandas() if os.path.exists(paths["channel_leaderboard"]) else pd.DataFrame(),
            "publish_hour_effect": spark.read.format("delta").load(paths["publish_hour_effect"]).toPandas() if os.path.exists(paths["publish_hour_effect"]) else pd.DataFrame(),
            "video_scoring": spark.read.format("delta").load(paths["video_scoring"]).toPandas() if os.path.exists(paths["video_scoring"]) else pd.DataFrame(),
        }
    except Exception:
        return None
    finally:
        spark.stop()
    return data


@st.cache_data
def load_search_delta_silver_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    silver_path = os.path.join(base_dir, "data", "processed", "silver", "search")
    if not os.path.exists(silver_path):
        return None

    spark = (
        SparkSession.builder
        .appName("SearchDeltaSilverDashboardReader")
        .master("local[*]")
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = spark.read.format("delta").load(silver_path).toPandas()
    except Exception:
        return None
    finally:
        spark.stop()
    return df


@st.cache_data
def load_trending_delta_gold_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gold_base = os.path.join(base_dir, "data", "analysis", "gold", "trending_analysis")
    paths = {
        "descriptive": os.path.join(gold_base, "descriptive"),
        "diagnostic": os.path.join(gold_base, "diagnostic"),
        "predictive": os.path.join(gold_base, "predictive"),
        "prescriptive": os.path.join(gold_base, "prescriptive"),
        "channel_leaderboard": os.path.join(gold_base, "channel_leaderboard"),
        "publish_hour_effect": os.path.join(gold_base, "publish_hour_effect"),
        "video_scoring": os.path.join(gold_base, "video_scoring"),
    }

    required = ["descriptive", "diagnostic", "predictive", "prescriptive"]
    if not all(os.path.exists(paths[k]) for k in required):
        return None

    spark = (
        SparkSession.builder
        .appName("TrendingDeltaGoldDashboardReader")
        .master("local[*]")
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        data = {
            "descriptive": spark.read.format("delta").load(paths["descriptive"]).toPandas(),
            "diagnostic": spark.read.format("delta").load(paths["diagnostic"]).toPandas(),
            "predictive": spark.read.format("delta").load(paths["predictive"]).toPandas(),
            "prescriptive": spark.read.format("delta").load(paths["prescriptive"]).toPandas(),
            "channel_leaderboard": spark.read.format("delta").load(paths["channel_leaderboard"]).toPandas() if os.path.exists(paths["channel_leaderboard"]) else pd.DataFrame(),
            "publish_hour_effect": spark.read.format("delta").load(paths["publish_hour_effect"]).toPandas() if os.path.exists(paths["publish_hour_effect"]) else pd.DataFrame(),
            "video_scoring": spark.read.format("delta").load(paths["video_scoring"]).toPandas() if os.path.exists(paths["video_scoring"]) else pd.DataFrame(),
        }
    except Exception:
        return None
    finally:
        spark.stop()
    return data


@st.cache_data
def load_trending_delta_silver_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    silver_path = os.path.join(base_dir, "data", "processed", "silver", "trending")
    if not os.path.exists(silver_path):
        return None

    spark = (
        SparkSession.builder
        .appName("TrendingDeltaSilverDashboardReader")
        .master("local[*]")
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = spark.read.format("delta").load(silver_path).toPandas()
    except Exception:
        return None
    finally:
        spark.stop()
    return df


df = load_comments_data()
search_df = load_search_data()
thumbnail_data = load_thumbnail_data()
thumbnail_delta_data = load_thumbnail_delta_data()
search_delta_gold = load_search_delta_gold_data()
search_delta_silver = load_search_delta_silver_data()
trending_delta_gold = load_trending_delta_gold_data()
trending_delta_silver = load_trending_delta_silver_data()

st.title("📹 YouTube Analytics Dashboard")
st.markdown("Comment sentiment and engagement, plus search-result video performance.")

if df is None and search_df is None:
    st.error("No processed data found. Run the processing pipeline first.")
    st.stop()

tab_comments, tab_search_delta, tab_trending, tab_thumbnail = st.tabs(
    ["Comments", "Search Analytics", "Trending", "Thumbnail"]
)

# =============================================================================
# COMMENTS TAB
# =============================================================================
with tab_comments:
    if df is None:
        st.warning("Comments data not found (`comments_processed.csv`).")
    else:
        st.sidebar.header("Comment filters")
        selected_video = st.sidebar.selectbox(
            "Filter by Video ID",
            ["All"] + list(df["video_id"].unique()),
            key="cmt_video",
        )
        selected_lang = st.sidebar.multiselect(
            "Filter by Language",
            options=df["language"].unique(),
            default=list(df["language"].unique()),
            key="cmt_lang",
        )

        filtered_df = df.copy()
        if selected_video != "All":
            filtered_df = filtered_df[filtered_df["video_id"] == selected_video]
        filtered_df = filtered_df[filtered_df["language"].isin(selected_lang)]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Comments", len(filtered_df))
        with col2:
            st.metric("Average Likes", round(filtered_df["like_count"].mean(), 2))
        with col3:
            pos_count = len(filtered_df[filtered_df["sentiment_label"] == "positive"])
            pct = round(pos_count / len(filtered_df) * 100, 1) if len(filtered_df) else 0
            st.metric("Positive Comments", f"{pos_count} ({pct}%)")
        with col4:
            st.metric("Unique Contributors", filtered_df["author"].nunique())

        st.divider()

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = filtered_df["sentiment_label"].value_counts()
            fig_sent = px.pie(
                names=sentiment_counts.index,
                values=sentiment_counts.values,
                color=sentiment_counts.index,
                color_discrete_map={
                    "positive": "#2ecc71",
                    "neutral": "#f1c40f",
                    "negative": "#e74c3c",
                },
                hole=0.4,
            )
            st.plotly_chart(fig_sent, use_container_width=True)

        with row1_col2:
            st.subheader("Top Languages")
            lang_counts = filtered_df["language"].value_counts().head(10)
            fig_lang = px.bar(
                x=lang_counts.index,
                y=lang_counts.values,
                labels={"x": "Language Code", "y": "Count"},
                color=lang_counts.values,
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_lang, use_container_width=True)

        st.divider()

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.subheader("Engagement: Likes vs Sentiment")
            fig_scatter = px.scatter(
                filtered_df,
                x="sentiment_score",
                y="like_count",
                color="sentiment_label",
                hover_data=["clean_text"],
                size="char_count",
                opacity=0.6,
                labels={
                    "sentiment_score": "Sentiment score",
                    "like_count": "Likes",
                    "sentiment_label": "Sentiment",
                    "char_count": "Comment length (characters)",
                },
                color_discrete_map={
                    "positive": "#2ecc71",
                    "neutral": "#f1c40f",
                    "negative": "#e74c3c",
                },
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with row2_col2:
            st.subheader("Comment Length Distribution")
            fig_hist = px.histogram(
                filtered_df,
                x="word_count",
                labels={"word_count": "Comment length (words)", "count": "Comments"},
                color_discrete_sequence=["#3498db"],
            )
            #TODO: Make bin size dynamic based on max word count (e.g. 20-word bins up to 200 words)
            fig_hist.update_traces(
                xbins=dict(
                    start=0,
                    end = 200,
                    size=10,
                )
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        st.subheader("Explore Comments")
        sort_by = st.selectbox(
            "Sort By",
            ["like_count", "sentiment_score", "word_count"],
            index=0,
            key="cmt_sort",
        )
        st.dataframe(
            filtered_df[
                [
                    "author",
                    "like_count",
                    "sentiment_label",
                    "sentiment_score",
                    "language",
                    "clean_text",
                ]
            ]
            .sort_values(by=sort_by, ascending=False)
            .head(50),
            use_container_width=True,
        )

#TODO: Remove this tab after testing of new search analytics using delta tables is complete.
# =============================================================================
# SEARCH ANALYTICS TAB
# =============================================================================
if False:  # search analytics hidden from UI (kept for reference)
    if search_df is None:
        st.warning("Search data not found (`search_processed.csv`).")
    else:
        st.sidebar.header("Search filters")
        cat_opts = sorted(search_df["category"].dropna().unique())
        sel_categories = st.sidebar.multiselect(
            "Categories",
            options=cat_opts,
            default=cat_opts,
            key="search_cat",
        )
        min_views = st.sidebar.number_input(
            "Minimum view count",
            min_value=0,
            value=0,
            step=10_000,
            key="search_min_views",
        )
        top_n_channels = st.sidebar.slider(
            "Top channels (chart)",
            min_value=5,
            max_value=25,
            value=12,
            key="search_top_n",
        )

        s = search_df.copy()
        if sel_categories:
            s = s[s["category"].isin(sel_categories)]
        s = s[s["view_count"] >= min_views]

        if len(s) == 0:
            st.warning("No videos match the current filters.")
        else:
            total_views = int(s["view_count"].sum())
            med_views = int(s["view_count"].median())
            avg_eng = float(s["engagement"].mean()) * 100
            med_dur_min = int(s["duration_sec"].median() // 60)
            med_dur_sec = int(s["duration_sec"].median() % 60)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Videos (filtered)", len(s))
            with m2:
                st.metric("Total views", f"{total_views:,}")
            with m3:
                st.metric("Avg engagement rate", f"{avg_eng:.2f}%")
            with m4:
                st.metric("Median duration", f"{med_dur_min}m {med_dur_sec}s")

            st.divider()

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Videos by category")
                cat_counts = s["category"].value_counts()
                fig_cat = px.bar(
                    x=cat_counts.values,
                    y=cat_counts.index,
                    orientation="h",
                    labels={"x": "Videos", "y": "Category"},
                    color=cat_counts.values,
                    color_continuous_scale="Blues",
                )
                fig_cat.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_cat, use_container_width=True)

            with c2:
                st.subheader("Category mix")
                fig_pie = px.pie(
                    names=cat_counts.index,
                    values=cat_counts.values,
                    hole=0.45,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.divider()

            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Views vs likes (log-scaled views)")
                fig_vl = px.scatter(
                    s,
                    x="view_count",
                    y="like_count",
                    color="category",
                    hover_data=["title", "channel_title"],
                    size="comment_count",
                    size_max=40,
                    opacity=0.75,
                    log_x=True,
                    labels={
                        "view_count": "Views",
                        "like_count": "Likes",
                        "category": "Category",
                        "comment_count": "Comments",
                    },
                )
                st.plotly_chart(fig_vl, use_container_width=True)

            with c4:
                st.subheader("View count distribution")
                fig_views = px.histogram(
                    s,
                    x="view_count",
                    nbins=40,
                    labels={"view_count": "Views", "count": "Videos"},
                    color_discrete_sequence=["#9b59b6"],
                )
                fig_views.update_xaxes(type="log", title="View count (log scale)")
                st.plotly_chart(fig_views, use_container_width=True)

            st.divider()

            c5, c6 = st.columns(2)
            with c5:
                st.subheader("Engagement rate distribution")
                fig_eng = px.histogram(
                    s,
                    x=s["engagement"] * 100,
                    nbins=35,
                    labels={"x": "Engagement rate (%)", "y": "Count"},
                    color_discrete_sequence=["#1abc9c"],
                )
                st.plotly_chart(fig_eng, use_container_width=True)

            with c6:
                st.subheader("Video age (days)")
                fig_age = px.histogram(
                    s,
                    x="video_age_days",
                    nbins=30,
                    labels={"video_age_days": "Video age (days)", "count": "Videos"},
                    color_discrete_sequence=["#e67e22"],
                )
                st.plotly_chart(fig_age, use_container_width=True)

            st.divider()

            st.subheader(f"Top {top_n_channels} channels by total views")
            ch = (
                s.groupby("channel_title", as_index=False)
                .agg(videos=("video_id", "count"), total_views=("view_count", "sum"))
                .sort_values("total_views", ascending=False)
                .head(top_n_channels)
            )
            fig_ch = px.bar(
                ch,
                x="total_views",
                y="channel_title",
                orientation="h",
                color="videos",
                labels={"total_views": "Total views", "channel_title": "Channel", "videos": "Videos"},
                color_continuous_scale="Viridis",
            )
            fig_ch.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_ch, use_container_width=True)

            st.divider()

            st.subheader("Search results table")
            sort_search = st.selectbox(
                "Sort by",
                [
                    "view_count",
                    "like_count",
                    "engagement",
                    "comment_count",
                    "video_age_days",
                    "duration_sec",
                ],
                key="search_tbl_sort",
            )
            display_cols = [
                "title",
                "channel_title",
                "category",
                "view_count",
                "like_count",
                "comment_count",
                "engagement",
                "duration_sec",
                "video_age_days",
                "published_at",
            ]
            show = [c for c in display_cols if c in s.columns]
            tbl = s[show].sort_values(by=sort_search, ascending=False)
            st.dataframe(tbl, use_container_width=True, hide_index=True)

# =============================================================================
# SEARCH DELTA GOLD TAB
# =============================================================================
with tab_search_delta:
    st.header("Search Analytics")
    if search_delta_silver is None:
        st.warning("Search Silver Delta data not found. Run `SearchDataProcessorDelta.py` first.")
    else:
        s = search_delta_silver.copy()
        for col in ("published_at", "fetched_at"):
            if col in s.columns:
                s[col] = pd.to_datetime(s[col], errors="coerce", utc=True)
        cat_opts_delta = sorted(s["category"].dropna().unique()) if "category" in s.columns else []
        sel_categories_delta = st.multiselect(
            "Categories",
            options=cat_opts_delta,
            default=cat_opts_delta,
            key="search_delta_cat",
        )
        min_views_delta = st.number_input(
            "Minimum view count",
            min_value=0,
            value=0,
            step=10_000,
            key="search_delta_min_views",
        )
        if sel_categories_delta:
            s = s[s["category"].isin(sel_categories_delta)]
        s = s[s["view_count"] >= min_views_delta]

        if len(s) == 0:
            st.warning("No videos match the current Delta filters.")
        else:
            total_views = int(s["view_count"].sum())
            avg_eng = float(s["engagement"].mean()) * 100
            med_dur_min = int(s["duration_sec"].median() // 60)
            med_dur_sec = int(s["duration_sec"].median() % 60)

            dm1, dm2, dm3, dm4 = st.columns(4)
            with dm1:
                st.metric("Total number of videos", len(s))
            with dm2:
                st.metric("Total views", f"{total_views:,}")
            with dm3:
                st.metric("Avg engagement rate", f"{avg_eng:.2f}%")
            with dm4:
                st.metric("Median duration", f"{med_dur_min}m {med_dur_sec}s")

            st.divider()

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Videos by category")
                cat_counts = s["category"].value_counts()
                fig_cat = px.bar(
                    x=cat_counts.values,
                    y=cat_counts.index,
                    orientation="h",
                    labels={"x": "Videos", "y": "Category"},
                    color=cat_counts.values,
                    color_continuous_scale="Blues",
                )
                fig_cat.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_cat, use_container_width=True)
            with c2:
                st.subheader("Category mix")
                fig_pie = px.pie(
                    names=cat_counts.index,
                    values=cat_counts.values,
                    hole=0.45,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.divider()
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Views vs likes")
                fig_vl = px.scatter(
                    s,
                    x="view_count",
                    y="like_count",
                    color="category",
                    hover_data=["title", "channel_title"],
                    size="comment_count",
                    size_max=40,
                    opacity=0.75,
                    log_x=True,
                    labels={
                        "view_count": "Views",
                        "like_count": "Likes",
                        "category": "Category",
                        "comment_count": "Comments",
                    },
                )
                st.plotly_chart(fig_vl, use_container_width=True)
            with c4:
                st.subheader("Video age (how much old)")
                fig_age = px.histogram(
                    s,
                    x="video_age_days",
                    nbins=30,
                    labels={"video_age_days": "Video age (days)", "count": "Videos"},
                    color_discrete_sequence=["#e67e22"],
                )
                st.plotly_chart(fig_age, use_container_width=True)

            if search_delta_gold is not None:
                descriptive_df = search_delta_gold.get("descriptive", pd.DataFrame())
                diagnostic_df = search_delta_gold.get("diagnostic", pd.DataFrame())
                predictive_df = search_delta_gold.get("predictive", pd.DataFrame())
                prescriptive_df = search_delta_gold.get("prescriptive", pd.DataFrame())
                query_df = search_delta_gold.get("query_performance", pd.DataFrame())
                channel_df = search_delta_gold.get("channel_leaderboard", pd.DataFrame())
                hour_df = search_delta_gold.get("publish_hour_effect", pd.DataFrame())
                score_df = search_delta_gold.get("video_scoring", pd.DataFrame())

                st.divider()
                st.subheader("**Descriptive analysis**")

                if not descriptive_df.empty and {"category", "avg_views"}.issubset(descriptive_df.columns):
                    st.caption("Shows category-level performance averages to summarize what is happening in the dataset.")
                    fig_desc_delta = px.bar(
                        descriptive_df.sort_values("avg_views", ascending=False).head(10),
                        x="category",
                        y="avg_views",
                        color="avg_engagement" if "avg_engagement" in descriptive_df.columns else None,
                        labels={"category": "Category", "avg_views": "Average views", "avg_engagement": "Average engagement"},
                    )
                    st.plotly_chart(fig_desc_delta, use_container_width=True)
                    st.dataframe(descriptive_df, use_container_width=True, hide_index=True)

                st.subheader("**Diagnostic analysis**")
                if not diagnostic_df.empty and {"metric_pair", "correlation"}.issubset(diagnostic_df.columns):
                    st.caption("Shows correlation strengths to explain which metrics move together and why performance patterns appear.")
                    diag_map = {
                        str(row["metric_pair"]): float(row["correlation"])
                        for _, row in diagnostic_df.iterrows()
                        if pd.notna(row["correlation"])
                    }
                    dd1, dd2, dd3 = st.columns(3)
                    with dd1:
                        st.metric(
                            "Views vs likes correlation",
                            f"{diag_map.get('view_count_like_count', 0.0):.3f}",
                        )
                    with dd2:
                        st.metric(
                            "Views vs comments correlation",
                            f"{diag_map.get('view_count_comment_count', 0.0):.3f}",
                        )
                    with dd3:
                        st.metric(
                            "Engagement vs duration correlation",
                            f"{diag_map.get('engagement_duration_sec', 0.0):.3f}",
                        )
                    st.dataframe(diagnostic_df, use_container_width=True, hide_index=True)

                st.subheader("**Predictive analysis**")
                if not predictive_df.empty and {"actual_view_count", "predicted_view_count"}.issubset(predictive_df.columns):
                    predictive_df = predictive_df.copy()
                    predictive_df["predicted_view_count"] = predictive_df["predicted_view_count"].clip(lower=0)
                    st.caption("Compares model-predicted views with actual views to estimate how well future performance can be forecasted.")
                    fig_pred_delta = px.scatter(
                        predictive_df,
                        x="actual_view_count",
                        y="predicted_view_count",
                        labels={
                            "actual_view_count": "Actual views",
                            "predicted_view_count": "Predicted views",
                        },
                        trendline="ols",
                    )
                    st.plotly_chart(fig_pred_delta, use_container_width=True)
                    st.dataframe(predictive_df.head(200), use_container_width=True, hide_index=True)

                st.subheader("**Prescriptive analysis**")
                if not prescriptive_df.empty:
                    st.caption("Highlights best-performing category strategies as actionable guidance for what to do next.")
                    if {"category", "best_avg_views"}.issubset(prescriptive_df.columns):
                        fig_presc_delta = px.bar(
                            prescriptive_df.sort_values("best_avg_views", ascending=False).head(12),
                            x="best_avg_views",
                            y="category",
                            orientation="h",
                            color="best_avg_engagement" if "best_avg_engagement" in prescriptive_df.columns else None,
                            labels={
                                "best_avg_views": "Best average views",
                                "category": "Category",
                                "best_avg_engagement": "Best average engagement",
                            },
                        )
                        fig_presc_delta.update_layout(yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig_presc_delta, use_container_width=True)
                    st.dataframe(prescriptive_df, use_container_width=True, hide_index=True)

                st.divider()
                st.subheader("Query performance")
                if not query_df.empty and {"search_query", "avg_views"}.issubset(query_df.columns):
                    fig_query = px.bar(
                        query_df.sort_values("avg_views", ascending=False).head(12),
                        x="search_query",
                        y="avg_views",
                        color="avg_engagement" if "avg_engagement" in query_df.columns else None,
                        labels={
                            "search_query": "Search query",
                            "avg_views": "Average views",
                            "avg_engagement": "Average engagement",
                        },
                    )
                    st.plotly_chart(fig_query, use_container_width=True)

                st.divider()
                st.subheader("Top channels by views")
                if not channel_df.empty and {"channel_title", "avg_views"}.issubset(channel_df.columns):
                    fig_channel = px.bar(
                        channel_df.sort_values("avg_views", ascending=False).head(12),
                        x="avg_views",
                        y="channel_title",
                        orientation="h",
                        color="videos" if "videos" in channel_df.columns else None,
                        labels={"avg_views": "Average views", "channel_title": "Channel", "videos": "Videos"},
                    )
                    fig_channel.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig_channel, use_container_width=True)

                st.subheader("Best publish hour")
                if not hour_df.empty and {"publish_hour", "avg_views"}.issubset(hour_df.columns):
                    hour_df_sorted = hour_df.sort_values("publish_hour").copy()
                    best_idx = hour_df_sorted["avg_views"].idxmax()
                    best_hour = int(hour_df_sorted.loc[best_idx, "publish_hour"])
                    best_avg_views = float(hour_df_sorted.loc[best_idx, "avg_views"])
                    st.metric("Best publish hour", f"{best_hour:02d}:00")
                    st.caption(f"Highest average views at this hour: {best_avg_views:,.0f}")

                    fig_hour = px.line(
                        hour_df_sorted,
                        x="publish_hour",
                        y="avg_views",
                        markers=True,
                        labels={"publish_hour": "Publish hour (24h)", "avg_views": "Average views"},
                    )
                    fig_hour.add_scatter(
                        x=[best_hour],
                        y=[best_avg_views],
                        mode="markers+text",
                        text=[f"Best: {best_hour:02d}:00"],
                        textposition="top center",
                        marker=dict(size=12, color="#e74c3c"),
                        name="Best hour",
                    )
                    fig_hour.update_xaxes(
                        title="Publish hour (24h)",
                    )
                    fig_hour.update_yaxes(title="Average views")
                    st.plotly_chart(fig_hour, use_container_width=True)

                st.divider()
                st.subheader("Top scored videos")
                if not score_df.empty:
                    st.caption("Score combines views, engagement, and freshness (higher score means better overall performance).")
                    cols = [c for c in ["title", "channel_title", "category", "search_query", "view_count", "engagement", "score"] if c in score_df.columns]
                    st.dataframe(score_df[cols].sort_values("score", ascending=False).head(25), use_container_width=True, hide_index=True)

                st.divider()
                st.subheader("Search results table")
                sort_search_delta = st.selectbox(
                    "Sort by",
                    [
                        "view_count",
                        "like_count",
                        "engagement",
                        "comment_count",
                        "video_age_days",
                        "duration_sec",
                    ],
                    key="search_delta_tbl_sort",
                )
                display_cols_delta = [
                    "title",
                    "channel_title",
                    "category",
                    "search_query",
                    "view_count",
                    "like_count",
                    "comment_count",
                    "engagement",
                    "duration_sec",
                    "video_age_days",
                    "published_at",
                ]
                show_delta = [c for c in display_cols_delta if c in s.columns]
                tbl_delta = s[show_delta].sort_values(by=sort_search_delta, ascending=False)
                st.dataframe(tbl_delta, use_container_width=True, hide_index=True)

# =============================================================================
# TRENDING TAB
# =============================================================================
with tab_trending:
    st.header("YouTube Trending Analytics")

    if trending_delta_silver is not None:
        t = trending_delta_silver.copy()
        for col in ("published_at", "fetched_at"):
            if col in t.columns:
                t[col] = pd.to_datetime(t[col], errors="coerce", utc=True)

        cat_opts = sorted(t["category"].dropna().unique()) if "category" in t.columns else []
        selected_categories = st.multiselect(
            "Categories",
            options=cat_opts,
            default=cat_opts,
            key="trending_delta_categories",
        )
        min_views = st.number_input(
            "Minimum view count",
            min_value=0,
            value=0,
            step=10_000,
            key="trending_delta_min_views",
        )

        if selected_categories:
            t = t[t["category"].isin(selected_categories)]
        if "view_count" in t.columns:
            t = t[t["view_count"] >= min_views]

        if t.empty:
            st.warning("No trending videos match the current filters.")
        else:
            total_views = int(t["view_count"].sum())
            avg_engagement = float(t["engagement"].mean()) * 100 if "engagement" in t.columns else 0.0
            median_likes = int(t["like_count"].median()) if "like_count" in t.columns else 0
            median_comments = int(t["comment_count"].median()) if "comment_count" in t.columns else 0

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Trending videos", len(t))
            with m2:
                st.metric("Total views", f"{total_views:,}")
            with m3:
                st.metric("Avg engagement rate", f"{avg_engagement:.2f}%")
            with m4:
                st.metric("Median likes/comments", f"{median_likes:,} / {median_comments:,}")

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Videos by category")
                cat_counts = t["category"].value_counts()
                fig_cat = px.bar(
                    x=cat_counts.values,
                    y=cat_counts.index,
                    orientation="h",
                    labels={"x": "Videos", "y": "Category"},
                    color=cat_counts.values,
                    color_continuous_scale="Blues",
                )
                fig_cat.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_cat, use_container_width=True)
            with col2:
                st.subheader("Category mix")
                fig_mix = px.pie(names=cat_counts.index, values=cat_counts.values, hole=0.45)
                st.plotly_chart(fig_mix, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Views vs likes")
                fig_vl = px.scatter(
                    t,
                    x="view_count",
                    y="like_count",
                    color="category",
                    hover_data=["title", "channel_title"],
                    size="comment_count",
                    size_max=40,
                    opacity=0.75,
                    log_x=True,
                )
                st.plotly_chart(fig_vl, use_container_width=True)
            with col4:
                st.subheader("Video age")
                fig_age = px.histogram(t, x="video_age_days", nbins=30)
                st.plotly_chart(fig_age, use_container_width=True)

            if trending_delta_gold is not None:
                descriptive_df = trending_delta_gold.get("descriptive", pd.DataFrame())
                diagnostic_df = trending_delta_gold.get("diagnostic", pd.DataFrame())
                predictive_df = trending_delta_gold.get("predictive", pd.DataFrame())
                prescriptive_df = trending_delta_gold.get("prescriptive", pd.DataFrame())
                channel_df = trending_delta_gold.get("channel_leaderboard", pd.DataFrame())
                hour_df = trending_delta_gold.get("publish_hour_effect", pd.DataFrame())
                score_df = trending_delta_gold.get("video_scoring", pd.DataFrame())

                st.divider()
                st.subheader("Descriptive analysis")
                if not descriptive_df.empty and {"category", "avg_views"}.issubset(descriptive_df.columns):
                    fig_desc = px.bar(
                        descriptive_df.sort_values("avg_views", ascending=False).head(12),
                        x="category",
                        y="avg_views",
                        color="avg_engagement" if "avg_engagement" in descriptive_df.columns else None,
                    )
                    fig_desc.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_desc, use_container_width=True)
                    st.dataframe(descriptive_df, use_container_width=True, hide_index=True)

                st.subheader("Diagnostic analysis")
                if not diagnostic_df.empty and {"metric_pair", "correlation"}.issubset(diagnostic_df.columns):
                    diag_map = {
                        str(row["metric_pair"]): float(row["correlation"])
                        for _, row in diagnostic_df.iterrows()
                        if pd.notna(row["correlation"])
                    }
                    d1, d2, d3, d4 = st.columns(4)
                    with d1:
                        st.metric("Views vs likes", f"{diag_map.get('view_count_like_count', 0.0):.3f}")
                    with d2:
                        st.metric("Views vs comments", f"{diag_map.get('view_count_comment_count', 0.0):.3f}")
                    with d3:
                        st.metric("Engagement vs duration", f"{diag_map.get('engagement_duration_sec', 0.0):.3f}")
                    with d4:
                        st.metric("Title length vs views", f"{diag_map.get('title_length_view_count', 0.0):.3f}")
                    st.dataframe(diagnostic_df, use_container_width=True, hide_index=True)

                st.subheader("Predictive analysis")
                if not predictive_df.empty and {"actual_like_count", "predicted_like_count"}.issubset(predictive_df.columns):
                    predictive_df = predictive_df.copy()
                    predictive_df["predicted_like_count"] = predictive_df["predicted_like_count"].clip(lower=0)
                    fig_pred = px.scatter(
                        predictive_df,
                        x="actual_like_count",
                        y="predicted_like_count",
                        trendline="ols",
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    st.dataframe(predictive_df.head(200), use_container_width=True, hide_index=True)

                st.subheader("Prescriptive analysis")
                if not prescriptive_df.empty:
                    if {"category", "best_avg_views"}.issubset(prescriptive_df.columns):
                        fig_presc = px.bar(
                            prescriptive_df.sort_values("best_avg_views", ascending=False).head(12),
                            x="best_avg_views",
                            y="category",
                            orientation="h",
                            color="best_avg_engagement" if "best_avg_engagement" in prescriptive_df.columns else None,
                        )
                        fig_presc.update_layout(yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig_presc, use_container_width=True)
                    st.dataframe(prescriptive_df, use_container_width=True, hide_index=True)

                st.divider()
                st.subheader("Top channels by average views")
                if not channel_df.empty and {"channel_title", "avg_views"}.issubset(channel_df.columns):
                    fig_channel = px.bar(
                        channel_df.sort_values("avg_views", ascending=False).head(12),
                        x="avg_views",
                        y="channel_title",
                        orientation="h",
                        color="videos" if "videos" in channel_df.columns else None,
                    )
                    fig_channel.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig_channel, use_container_width=True)

                st.subheader("Best publish hour")
                if not hour_df.empty and {"publish_hour", "avg_views"}.issubset(hour_df.columns):
                    hour_df_sorted = hour_df.sort_values("publish_hour").copy()
                    best_idx = hour_df_sorted["avg_views"].idxmax()
                    best_hour = int(hour_df_sorted.loc[best_idx, "publish_hour"])
                    best_avg_views = float(hour_df_sorted.loc[best_idx, "avg_views"])
                    st.metric("Best publish hour", f"{best_hour:02d}:00")
                    st.caption(f"Highest average views at this hour: {best_avg_views:,.0f}")
                    fig_hour = px.line(hour_df_sorted, x="publish_hour", y="avg_views", markers=True)
                    fig_hour.add_scatter(
                        x=[best_hour],
                        y=[best_avg_views],
                        mode="markers+text",
                        text=[f"Best: {best_hour:02d}:00"],
                        textposition="top center",
                        marker=dict(size=12, color="#e74c3c"),
                        name="Best hour",
                    )
                    st.plotly_chart(fig_hour, use_container_width=True)

                st.divider()
                st.subheader("Top scored trending videos")
                if not score_df.empty:
                    cols = [c for c in ["title", "channel_title", "category", "view_count", "like_count", "engagement", "score"] if c in score_df.columns]
                    st.dataframe(score_df[cols].sort_values("score", ascending=False).head(25), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Trending videos table")
            sort_col = st.selectbox(
                "Sort by",
                ["view_count", "like_count", "engagement", "comment_count", "video_age_days", "duration_sec"],
                key="trending_delta_sort",
            )
            display_cols = [
                "title",
                "channel_title",
                "category",
                "view_count",
                "like_count",
                "comment_count",
                "engagement",
                "duration_sec",
                "video_age_days",
                "published_at",
            ]
            show_cols = [c for c in display_cols if c in t.columns]
            st.dataframe(t[show_cols].sort_values(by=sort_col, ascending=False), use_container_width=True, hide_index=True)

    else:
        st.warning("Trending Silver Delta data not found. Run `TrendingDataProcessorDelta.py` first.")

# =============================================================================
# THUMBNAIL ANALYTICS TAB
# =============================================================================
with tab_thumbnail:
    st.header("YouTube Thumbnail Analytics")
    
    # Original Analytics from JSON
    if thumbnail_data is None:
        st.warning("Original thumbnail analytics data not found (`final_analytics_report.json`).")
    else:
        analytics = thumbnail_data
        #st.subheader("📊 Original Analytics (from JSON)")

        # Level 1: Descriptive Analytics
        st.markdown("**Descriptive Analytics: Category Performance**")
        desc_df = pd.DataFrame(analytics["level_1_descriptive"])
        col1, col2 = st.columns(2)
        with col1:
            fig_thumb_views = px.bar(
                desc_df,
                x="category",
                y="avg_views",
                title="Average Views by Category",
                color="avg_views",
                labels={"category": "Category", "avg_views": "Average views"},
                color_continuous_scale="Blues",
            )
            fig_thumb_views.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_thumb_views, use_container_width=True)
        with col2:
            fig_thumb_videos = px.bar(
                desc_df,
                x="category",
                y="total_videos",
                title="Total Videos by Category",
                color="total_videos",
                labels={"category": "Category", "total_videos": "Total videos"},
                color_continuous_scale="Viridis",
            )
            fig_thumb_videos.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_thumb_videos, use_container_width=True)
        st.dataframe(desc_df, use_container_width=True)

        st.divider()

        # Level 2: Diagnostic Analytics
        st.markdown("**Diagnostic Analytics: Correlations**")
        diag = analytics["level_2_diagnostic"]
        col3, col4 = st.columns(2)
        with col3:
            st.metric("View-Like Correlation", f"{diag['view_like_correlation']:.3f}")
        with col4:
            st.metric("Thumbnail-Clickbait Correlation", f"{diag['clickbait_thumbnail_correlation']:.3f}")
        st.markdown(
            f"**Insight:** The thumbnail analysis shows a strong relationship between views and likes, while thumbnail clickbait correlation is {diag['clickbait_thumbnail_correlation']:.3f}."
        )

        st.divider()

        # Level 3: Predictive Analytics
        st.markdown("**Predictive Analytics: Like Count Predictions**")
        pred_df = pd.DataFrame(analytics["level_3_predictive"])
        if not pred_df.empty:
            fig_thumb_pred = px.scatter(
                pred_df,
                x="view_count_feature",
                y="like_count",
                size="prediction",
                hover_data=["prediction"],
                title="Views vs Actual Likes",
                labels={"view_count_feature": "View Count Feature", "like_count": "Actual Likes"},
            )
            st.plotly_chart(fig_thumb_pred, use_container_width=True)
            st.dataframe(pred_df, use_container_width=True)

        st.divider()

        # Level 4: Prescriptive Analytics
        st.markdown("**Prescriptive Analytics: Thumbnail Recommendations**")
        presc = analytics["level_4_prescriptive"]
        st.metric("Optimal Upload Hour", f"{presc['optimal_hour']}:00")
        st.info(presc["action"])

        st.divider()

        # Big data metrics
        st.markdown("**Big Data Metrics**")
        metrics = analytics.get("big_data_metrics", {})
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Total Records Processed", f"{metrics.get('total_records_processed', 'N/A')}")
        with m2:
            st.metric("Thumbnail Data Points", f"{metrics.get('thumbnail_data_points', 'N/A')}")

    st.divider()

    # New Analytics from Delta Table
    if thumbnail_delta_data is None:
        st.warning("New thumbnail analytics data not found (Silver Delta table).")
    else:
        analytics = thumbnail_delta_data["thumbnail_analysis"]
        # st.subheader("🆕 New Analytics (from Delta Table)")

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Thumbnails", analytics["total_thumbnails"])
        with col2:
            st.metric("Avg Brightness", f"{analytics['avg_brightness']:.2f}")
        with col3:
            st.metric("Avg Contrast", f"{analytics['avg_contrast']:.2f}")
        with col4:
            st.metric("Avg Quality Score", f"{analytics['avg_quality_score']:.2f}")

        st.divider()

        # Detailed Metrics
        st.markdown("**Thumbnail Quality Metrics**")
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Avg Colorfulness", f"{analytics['avg_colorfulness']:.2f}")
        with col6:
            st.metric("Avg Sharpness", f"{analytics['avg_sharpness']:.2f}")

        st.divider()

        # Brightness Distribution
        st.markdown("**Brightness Distribution**")
        brightness_stats = analytics["brightness_distribution"]
        st.write(f"**Mean:** {brightness_stats['mean']:.2f}")
        st.write(f"**Std:** {brightness_stats['std']:.2f}")
        st.write(f"**Min:** {brightness_stats['min']:.2f}")
        st.write(f"**Max:** {brightness_stats['max']:.2f}")

        st.divider()

        # Top Dominant Colors
        st.markdown("**Top Dominant Colors**")
        colors_df = pd.DataFrame(list(analytics["top_dominant_colors"].items()), columns=["Color", "Count"])
        fig_colors = px.bar(
            colors_df,
            x="Color",
            y="Count",
            title="Most Common Dominant Colors in Thumbnails",
            color="Count",
            color_continuous_scale="Rainbow",
        )
        st.plotly_chart(fig_colors, use_container_width=True)
        st.dataframe(colors_df, use_container_width=True)

st.markdown("---")
st.caption("Dashboard generated by Antigravity Dashboard Engine. Run with: `streamlit run src/Dashboard/app_dashboard.py`")
