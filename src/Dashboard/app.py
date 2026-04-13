import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json

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
def load_trending_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "processed", "trending_processed.json")
    if not os.path.exists(data_path):
        return None
    with open(data_path, "r") as f:
        return json.load(f)


df = load_comments_data()
search_df = load_search_data()
trending_data = load_trending_data()

st.title("📹 YouTube Analytics Dashboard")
st.markdown("Comment sentiment and engagement, plus search-result video performance.")

if df is None and search_df is None:
    st.error("No processed data found. Run the processing pipeline first.")
    st.stop()

tab_comments, tab_search, tab_trending = st.tabs(["Comments", "Search analytics", "Trending"])

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
                nbins=30,
                color_discrete_sequence=["#3498db"],
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

# =============================================================================
# SEARCH ANALYTICS TAB
# =============================================================================
with tab_search:
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
                )
                st.plotly_chart(fig_vl, use_container_width=True)

            with c4:
                st.subheader("View count distribution")
                fig_views = px.histogram(
                    s,
                    x="view_count",
                    nbins=40,
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
                labels={"total_views": "Total views", "channel_title": "Channel"},
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
                "tag_count",
                "video_age_days",
                "published_at",
            ]
            show = [c for c in display_cols if c in s.columns]
            tbl = s[show].sort_values(by=sort_search, ascending=False)
            st.dataframe(tbl, use_container_width=True, hide_index=True)

# =============================================================================
# TRENDING TAB
# =============================================================================
with tab_trending:
    if trending_data is None:
        st.warning("Trending data not found (`trending_processed.json`).")
    else:
        analytics = trending_data["analytics_results"]
        
        st.header("YouTube Trending Analytics")
        
        # Level 1: Descriptive Analytics
        st.subheader("📊 Descriptive Analytics: Category Performance")
        desc_df = pd.DataFrame(analytics["level_1_descriptive"])
        col1, col2 = st.columns(2)
        with col1:
            fig_desc_views = px.bar(
                desc_df,
                x="category",
                y="avg_views",
                title="Average Views by Category",
                color="avg_views",
                color_continuous_scale="Blues",
            )
            fig_desc_views.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_desc_views, use_container_width=True)
        with col2:
            fig_desc_likes = px.bar(
                desc_df,
                x="category",
                y="avg_likes",
                title="Average Likes by Category",
                color="avg_likes",
                color_continuous_scale="Greens",
            )
            fig_desc_likes.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_desc_likes, use_container_width=True)
        
        st.dataframe(desc_df, use_container_width=True)
        
        st.divider()
        
        # Level 2: Diagnostic Analytics
        st.subheader("🔍 Diagnostic Analytics: Correlations")
        corr = analytics["level_2_diagnostic"]["correlation_views_likes"]
        st.metric("Correlation between Views and Likes", f"{corr:.3f}")
        st.markdown(f"**Insight:** Views and likes have a moderate positive correlation of {corr:.3f}.")
        
        st.divider()
        
        # Level 3: Predictive Analytics
        st.subheader("🔮 Predictive Analytics: Like Count Predictions")
        pred_df = pd.DataFrame(analytics["level_3_predictive"])
        if not pred_df.empty:
            pred_df["features"] = pred_df["features"].apply(lambda x: x[0] if isinstance(x, list) and x else x)
            fig_pred = px.scatter(
                pred_df,
                x="like_count",
                y="prediction",
                title="Actual vs Predicted Likes",
                labels={"like_count": "Actual Likes", "prediction": "Predicted Likes"},
                trendline="ols",
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            st.dataframe(pred_df, use_container_width=True)
        
        st.divider()
        
        # Level 4: Prescriptive Analytics
        st.subheader("💡 Prescriptive Analytics: Publishing Recommendations")
        presc = analytics["level_4_prescriptive"]
        st.metric("Best Publishing Hour", f"{presc['best_publish_hour']}:00")
        st.info(presc["action"])
        
        hourly_df = pd.DataFrame(presc["hourly_trends"])
        fig_hourly = px.bar(
            hourly_df,
            x="publish_hour",
            y="avg_views",
            title="Average Views by Publishing Hour",
            color="avg_views",
            color_continuous_scale="Oranges",
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
        st.dataframe(hourly_df, use_container_width=True)

st.markdown("---")
st.caption("Dashboard generated by Antigravity Dashboard Engine. Run with: `streamlit run src/Dashboard/app_dashboard.py`")
