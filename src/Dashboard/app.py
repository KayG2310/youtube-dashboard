import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path

st.set_page_config(page_title="YouTube Trending Analytics", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[2]
GOLD     = BASE_DIR / "data" / "delta_lake" / "gold"
NEW_DATA = BASE_DIR / "new_data"

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

# Representative dirs per grouped category for search/thumbnail images
CATEGORY_DIRS = {
    "Education":     NEW_DATA / "Education",
    "Gaming":        NEW_DATA / "Gaming",
    "Entertainment": NEW_DATA / "entertainment",
    "Music":         NEW_DATA / "music",
    "News":          NEW_DATA / "politics",
    "Sports":        NEW_DATA / "sports",
}

PALETTE = ["#0080FF", "#FF5C5C", "#00D1B2", "#9B59B6", "#F1C40F", "#34495E"]


# ── helpers ──────────────────────────────────────────────────────────────────
@st.cache_data
def read_gold(subpath: str) -> pd.DataFrame:
    p = GOLD / subpath
    try:
        import deltalake as dl
        return dl.DeltaTable(str(p)).to_pandas()
    except Exception:
        pq = p / "_data.parquet"
        return pd.read_parquet(str(pq)) if pq.exists() else pd.DataFrame()

@st.cache_data
def load_raw_trending():
    df = pd.read_csv(NEW_DATA / "US_Trending_filtered.csv",
                     parse_dates=["trending_date"],
                     dayfirst=False)
    df["category"] = df["category"].map(CATEGORY_MAP).fillna(df["category"])
    return df

@st.cache_data
def load_raw_comments():
    return pd.read_csv(NEW_DATA / "US_Trending_Comments.csv")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stage-header {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    color: #1a1a2e; border-radius: 14px; padding: 22px 30px; margin-bottom: 25px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.stage-header h2 { margin:0; font-size:1.8rem; font-weight: 700; }
.stage-header p  { margin:6px 0 0; opacity:.8; font-size:1rem; }
.kpi-card {
    background: white;
    border: 1px solid #e1e4e8;
    border-radius: 12px; padding: 18px 22px; color: #1a1a2e; text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.kpi-val { font-size: 2.2rem; font-weight: 700; color: #0080FF; }
.kpi-lbl { font-size: .9rem; opacity: .7; margin-top: 4px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

def stage_header(title, subtitle):
    st.markdown(f'<div class="stage-header"><h2>{title}</h2><p>{subtitle}</p></div>',
                unsafe_allow_html=True)

def kpi(col, value, label):
    col.markdown(f'<div class="kpi-card"><div class="kpi-val">{value}</div>'
                 f'<div class="kpi-lbl">{label}</div></div>', unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg", width=130)
st.sidebar.title("YouTube Trending")
stage = st.sidebar.radio("Analytics Stage", [
    "1 · Descriptive", "2 · Diagnostic", "3 · Predictive", "4 · Prescriptive"])
st.sidebar.markdown("---")
# st.sidebar.caption("Pipeline: Kafka → Bronze → Gold (Delta Lake)")

raw = load_raw_trending()
raw["trending_date"] = pd.to_datetime(raw["trending_date"], errors="coerce", dayfirst=False)
raw["engagement_rate"] = ((raw["likes"] + raw["comments"]) / raw["views"].replace(0, 1)).round(4)
raw["trending_month"] = raw["trending_date"].dt.month
raw["trending_quarter"] = raw["trending_date"].dt.quarter

# =============================================================================
# STAGE 1 – DESCRIPTIVE
# =============================================================================
if stage == "1 · Descriptive":
    stage_header("Descriptive Analytics", "What is happening? — category volume, sentiment distribution, thumbnail stats")

    cat_vol   = read_gold("descriptive/category_volume")
    monthly   = read_gold("descriptive/monthly_category_counts")
    cat_sent  = read_gold("descriptive/category_sentiment")
    quarterly = read_gold("descriptive/quarterly_category_counts")

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    kpi(k1, f"{len(raw):,}", "Total Trending Videos")
    kpi(k2, raw['category'].nunique(), "Categories")
    kpi(k3, f"{raw['views'].sum()/1e9:.1f}B", "Total Views")
    kpi(k4, f"{raw['engagement_rate'].mean()*100:.2f}%", "Avg Engagement")
    st.markdown("<br>", unsafe_allow_html=True)

    # Key Insights Callout
    # st.markdown("""
    # <div style="background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; border-left: 5px solid #FF6584; margin-bottom: 25px;">
    #     <span style="font-size: 1.1rem; font-weight: 600;">Key Observations</span><br>
    #     <p style="margin: 5px 0 0 0; opacity: 0.9;">
    #         While <b>Music</b> maintains the highest average engagement rate, <b>Education</b> videos consistently receive the highest number of average likes.
    #     </p>
    # </div>
    # """, unsafe_allow_html=True)

    # Category volume bar
    if not cat_vol.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Category Average Views")
            fig = px.bar(cat_vol.sort_values("avg_views", ascending=True),
                         x="avg_views", y="category", orientation="h",
                         color="avg_views", color_continuous_scale="Blues",
                         labels={"avg_views":"Average Views","category":""})
            fig.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Category Share (by video count)")
            fig2 = px.pie(cat_vol, names="category", values="video_count",
                          color_discrete_sequence=PALETTE, hole=0.45)
            fig2.update_traces(textposition="inside", textinfo="percent+label")
            fig2.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # Monthly area chart
    if not monthly.empty:
        st.subheader("Monthly Trending Counts per Category")
        import calendar
        monthly["month_name"] = monthly["trending_month"].apply(lambda m: calendar.month_abbr[int(m)])
        fig_hm = px.area(monthly.sort_values(["trending_month", "category"]),
                         x="month_name", y="video_count", color="category",
                         color_discrete_sequence=PALETTE,
                         labels={"month_name": "Month", "video_count": "Videos", "category": ""})
        fig_hm.update_layout(height=360)
        st.plotly_chart(fig_hm, use_container_width=True)

    # Sentiment: stacked % bar + VADER score diverging bar
    if not cat_sent.empty:
        st.subheader("Comment Sentiment per Category (VADER scores)")
        st.caption("VADER compound scores range from −1 (very negative) to +1 (very positive). Great for emoji/slang-heavy YouTube comments.")
        c3, c4 = st.columns(2)
        with c3:
            # 100% stacked bar so categories with different comment volumes are comparable
            total = cat_sent.groupby("category")["comment_count"].transform("sum")
            cat_sent = cat_sent.copy()
            cat_sent["pct"] = (cat_sent["comment_count"] / total * 100).round(1)
            fig_sent = px.bar(cat_sent, x="pct", y="category", color="sentiment_label",
                              orientation="h", barmode="stack",
                              color_discrete_map={"positive":"#27ae60","neutral":"#f39c12","negative":"#c0392b"},
                              labels={"pct":"% of Comments","sentiment_label":"Sentiment","category":""},
                              title="Sentiment Mix (%)")
            fig_sent.update_layout(height=380)
            st.plotly_chart(fig_sent, use_container_width=False)
        # with c4:
        #     # Diverging bar: avg VADER score, centered at 0
        #     avg_s = cat_sent.groupby("category")["avg_score"].mean().reset_index()
        #     avg_s = avg_s.sort_values("avg_score")
        #     avg_s["color"] = avg_s["avg_score"].apply(lambda v: "#27ae60" if v >= 0 else "#c0392b")
        #     fig_s2 = go.Figure(go.Bar(
        #         x=avg_s["avg_score"], y=avg_s["category"],
        #         orientation="h",
        #         marker_color=avg_s["color"],
        #         text=avg_s["avg_score"].round(3),
        #         textposition="outside"
        #     ))
        #     fig_s2.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        #     fig_s2.update_layout(height=380, xaxis_title="Avg VADER Compound Score",
        #                           title="Avg Sentiment Score by Category",
        #                           xaxis_range=[-0.5, 0.5])
        #     st.plotly_chart(fig_s2, use_container_width=True)

    # Engagement funnel: views → likes → comments per category
    if not cat_vol.empty:
        st.subheader("Engagement :  Likes & Comments")
        st.caption("Music videos often lead in engagement percentage, but Education content drives the highest absolute average like counts.")
        funnel_data = cat_vol.sort_values("avg_views", ascending=False)
        fig_funnel = go.Figure()
        # fig_funnel.add_trace(go.Bar(name="Avg Views", x=funnel_data["category"],
        #                             y=funnel_data["avg_views"], marker_color="#6C63FF"))
        fig_funnel.add_trace(go.Bar(name="Avg Likes", x=funnel_data["category"],
                                    y=funnel_data["avg_likes"], marker_color="#FF6584"))
        fig_funnel.add_trace(go.Bar(name="Avg Comments", x=funnel_data["category"],
                                    y=funnel_data["avg_comments"], marker_color="#43BCCD"))
        fig_funnel.update_layout(barmode="group", height=380,
                                  xaxis_tickangle=-30, yaxis_title="Count")
        st.plotly_chart(fig_funnel, use_container_width=True)

    # Thumbnail overview images
    st.subheader("Thumbnail Attribute Averages")
    img1 = NEW_DATA / "Thumbnail_Attributes_Averages.jpeg"
    img2 = NEW_DATA / "Thumbnail_Engagement_Averages.jpeg"
    tc1, tc2 = st.columns(2)
    if img1.exists(): tc1.image(str(img1), caption="Attribute Averages", use_column_width=False)
    # if img2.exists(): tc2.image(str(img2), caption="Engagement Averages", use_column_width=True)

# =============================================================================
# STAGE 2 – DIAGNOSTIC
# =============================================================================
elif stage == "2 · Diagnostic":
    stage_header("Diagnostic Analytics",
                 "Why did it happen? — search trends, sentiment drivers, thumbnail correlations")

    heatmap_df = read_gold("diagnostic/month_category_heatmap")
    sent_eng   = read_gold("diagnostic/sentiment_vs_engagement")
    sent_stack = read_gold("diagnostic/sentiment_stack")
    thumb_corr = read_gold("diagnostic/thumbnail_category_correlation")

    # Month × category grouped bar
    if not heatmap_df.empty:
        st.subheader("When Each Category Trended (Monthly Breakdown)")
        import calendar
        heatmap_df["month_name"] = heatmap_df["trending_month"].apply(lambda m: calendar.month_abbr[int(m)])
        fig_h = px.bar(heatmap_df.sort_values(["trending_month", "category"]),
                       x="month_name", y="video_count", color="category", barmode="group",
                       color_discrete_sequence=PALETTE,
                       labels={"month_name": "Month", "video_count": "Video Count", "category": ""})
        fig_h.update_layout(height=380)
        st.plotly_chart(fig_h, use_container_width=True)

    # Search trends per category
    st.subheader(" Search Trends per Category")
    cats_with_search = {k: v for k, v in CATEGORY_DIRS.items() if (v / "search.jpeg").exists()}
    if cats_with_search:
        cols_per_row = 3
        items = list(cats_with_search.items())
        for row_start in range(0, len(items), cols_per_row):
            row_items = items[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, (cat, path) in zip(cols, row_items):
                col.image(str(path / "search.jpeg"), caption=f"{cat} — Search Trend", use_column_width=True)

    # Sentiment vs engagement scatter
    if not sent_eng.empty:
        st.subheader("Comment Sentiment vs Category Engagement")
        st.caption("Music not only leads in engagement rate but also carries the most positive sentiment (VADER 0.21), suggesting a highly receptive audience.")
        c1, c2 = st.columns(2)
        with c1:
            fig_se = px.scatter(sent_eng, x="avg_sentiment", y="avg_engagement",
                                size="total_comments", color="category",
                                color_discrete_sequence=PALETTE,
                                hover_data=["pct_positive","pct_negative","avg_views"],
                                labels={"avg_sentiment":"Avg Comment Sentiment",
                                        "avg_engagement":"Avg Engagement Rate",
                                        "total_comments":"# Comments"})
            fig_se.update_layout(height=380)
            st.plotly_chart(fig_se, use_container_width=True)
        with c2:
            # pct positive/negative grouped bar
            melt = sent_eng[["category","pct_positive","pct_negative"]].melt(
                id_vars="category", var_name="type", value_name="pct")
            melt["type"] = melt["type"].map({"pct_positive":"Positive","pct_negative":"Negative"})
            fig_pct = px.bar(melt, x="category", y="pct", color="type", barmode="group",
                             color_discrete_map={"Positive":"#2ecc71","Negative":"#e74c3c"},
                             labels={"pct":"Fraction of Comments","category":""})
            fig_pct.update_layout(height=380, xaxis_tickangle=-30)
            st.plotly_chart(fig_pct, use_container_width=True)

    # Thumbnail correlation heatmap
    if not thumb_corr.empty:
        st.subheader("Thumbnail Visual Attributes vs Engagement")
        c3, c4 = st.columns(2)
        with c3:
            # Radar per category
            cats = thumb_corr["category"].tolist()
            fig_radar = go.Figure()
            metrics = ["brightness_avg","contrast_avg","colorfulness_avg"]
            labels  = ["Brightness","Contrast","Colorfulness"]
            for _, row in thumb_corr.iterrows():
                vals = [row[m] for m in metrics] + [row[metrics[0]]]
                fig_radar.add_trace(go.Scatterpolar(r=vals, theta=labels+[labels[0]],
                                                    fill="toself", name=row["category"]))
            fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0,1])),
                                    height=420, legend=dict(font=dict(size=9)))
            st.plotly_chart(fig_radar, use_container_width=False)
        # with c4:
        #     fig_tc = px.scatter(thumb_corr, x="colorfulness_avg", y="avg_engagement",
        #                         size="contrast_avg", color="category",
        #                         color_discrete_sequence=PALETTE,
        #                         labels={"colorfulness_avg":"Colorfulness","avg_engagement":"Avg Engagement"},
        #                         hover_data=["brightness_avg","contrast_avg"])
        #     fig_tc.update_layout(height=420)
        #     st.plotly_chart(fig_tc, use_container_width=True)

    # Per-category thumbnail scatter images
    st.subheader("Category Thumbnail Engagement Scatter Plots")
    thumb_img_cats = {k: v for k, v in CATEGORY_DIRS.items()
                      if (v / "thumbnail_engagement_scatter.jpeg").exists()}
    if thumb_img_cats:
        selected_cat = st.selectbox("Select Category to View Scatter Analysis", list(thumb_img_cats.keys()))
        if selected_cat:
            img_path = thumb_img_cats[selected_cat] / "thumbnail_engagement_scatter.jpeg"
            st.image(str(img_path), caption=f"Thumbnail Engagement Analysis: {selected_cat}", use_column_width=True)

# =============================================================================
# STAGE 3 – PREDICTIVE
# =============================================================================
elif stage == "3 · Predictive":
    stage_header("Predictive Analytics",
                 "")

    import calendar as _cal
    forecast = read_gold("predictive/category_forecast")
    prob_df  = read_gold("predictive/category_trend_probability")
    norm_vol = read_gold("predictive/monthly_norm_volume")
    model_fit = read_gold("predictive/model_fit")

    # ── Model fit: actual vs predicted ──────────────────────────────────────
    if not model_fit.empty and "predicted_video_count" in model_fit.columns:
        st.subheader("Ridge Model — Actual vs Predicted (training data)")
        st.caption("Compares Ridge model predictions against actual training data over time. Closer lines = better fit.")
        melted_fit = model_fit.rename(columns={"video_count": "Actual", "predicted_video_count": "Predicted"}).melt(
            id_vars=["category", "trending_month"], value_vars=["Actual", "Predicted"],
            var_name="Type", value_name="Count"
        )
        import calendar
        melted_fit["Month"] = melted_fit["trending_month"].apply(lambda m: calendar.month_abbr[int(m)])

        fig_fit = px.line(melted_fit.sort_values(["category", "trending_month"]),
                          x="Month", y="Count", color="Type",
                          facet_col="category", facet_col_wrap=3,
                          markers=True,
                          color_discrete_map={"Actual": "#6C63FF", "Predicted": "#FF6584"})
        
        # Clean up facet annotations (remove "category=" prefix)
        fig_fit.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig_fit.update_layout(height=450, legend_title="")
        st.plotly_chart(fig_fit, use_container_width=True)

    # ── Confidence band forecast ─────────────────────────────────────────────
    if not forecast.empty and "ci_lower" in forecast.columns:
        st.subheader("3-Month Forecast with Confidence Bands — May · Jun · Jul 2026")
        # st.caption("Shaded area = ±1.5σ confidence band from Ridge regression residuals. Higher = more predicted trending videos.")

        # months = sorted(forecast["forecast_month"].unique(),
        #                 key=lambda x: ["May 2026","June 2026","July 2026"].index(x) if x in ["May 2026","June 2026","July 2026"] else 99)
        # cats   = forecast["category"].unique()
        # colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cats)}

        # # One chart per forecast month showing CI bars
        # tab_cols = st.columns(len(months))
        # for col, m in zip(tab_cols, months):
        #     sub = forecast[forecast["forecast_month"] == m].sort_values(
        #         "predicted_video_count", ascending=True)
        #     fig_ci = go.Figure()
        #     fig_ci.add_trace(go.Bar(
        #         x=sub["predicted_video_count"], y=sub["category"],
        #         orientation="h",
        #         marker_color=[colors[c] for c in sub["category"]],
        #         error_x=dict(
        #             type="data",
        #             symmetric=False,
        #             array=(sub["ci_upper"] - sub["predicted_video_count"]).tolist(),
        #             arrayminus=(sub["predicted_video_count"] - sub["ci_lower"]).tolist(),
        #             color="rgba(255,255,255,0.6)"
            #     ),
            #     name=m
            # ))
            # fig_ci.update_layout(height=360, title=m, showlegend=False,
            #                       xaxis_title="Predicted Videos")
            # col.plotly_chart(fig_ci, use_container_width=True)

        # Annual events callouts
        st.subheader("Annual Events Influencing Forecasts")
        event_rows = forecast[forecast["annual_event"] != ""][["forecast_month","category","annual_event","predicted_video_count"]]
        if not event_rows.empty:
            ev_cols = st.columns(3)
            for i, (_, r) in enumerate(event_rows.iterrows()):
                ev_cols[i % 3].info(f"**{r['forecast_month']}** · {r['category']}\n\n_{r['annual_event']}_\n\nPredicted: **{int(r['predicted_video_count'])}** videos")

        # Summary forecast line chart
        st.subheader("Forecast Trendline")
        fig_fhm = px.line(forecast.sort_values(["month_num", "category"]),
                          x="forecast_month", y="predicted_video_count", color="category",
                          markers=True, color_discrete_sequence=PALETTE,
                          labels={"forecast_month": "Month", "predicted_video_count": "Predicted Videos", "category": ""})
        fig_fhm.update_layout(height=360)
        st.plotly_chart(fig_fhm, use_container_width=True)

# =============================================================================
# STAGE 4 – PRESCRIPTIVE
# =============================================================================
elif stage == "4 · Prescriptive":
    stage_header("Prescriptive Analytics",
                 "")

    opp      = read_gold("prescriptive/category_opportunity")
    recs     = read_gold("prescriptive/thumbnail_recommendations")
    best_m   = read_gold("prescriptive/best_posting_month")

    if not opp.empty:
        # Top-3 callouts first — most actionable
        top3 = opp.head(3)
        st.markdown("### Top Categories to Target Right Now")
        cols_t = st.columns(3)
        medals = ["1.","2.","3."]
        for col, medal, (_, row) in zip(cols_t, medals, top3.iterrows()):
            col.success(f"{medal} **{row['category']}**\n\n"
                        f"Opportunity Score: **{row['opportunity_score']:.2f}**\n\n"
                        f"Avg Engagement: **{row['avg_engagement']*100:.2f}%**\n\n"
                        f"Avg Views: **{int(row['avg_views']):,}**")
        
        st.info("Strategic Note: While Music is the clear leader for viral engagement, Education offers a high-retention 'Liking' audience that is more consistent than Entertainment or News.")

        st.subheader("Full Category Opportunity Landscape")
        st.caption("Opportunity = (engagement rank + volume rank) / 2. Bubble size = video count.")
        c1, c2 = st.columns(2)
        with c1:
            fig_opp = px.bar(opp.sort_values("opportunity_score", ascending=True),
                             x="opportunity_score", y="category", orientation="h",
                             color="opportunity_score", color_continuous_scale="Viridis",
                             labels={"opportunity_score":"Opportunity Score","category":""})
            fig_opp.add_vline(x=0.5, line_dash="dash", line_color="white", opacity=0.4,
                              annotation_text="Threshold")
            fig_opp.update_layout(height=380)
            st.plotly_chart(fig_opp, use_container_width=False)
        # with c2:
        #     fig_opp2 = px.scatter(opp, x="avg_engagement", y="avg_views",
        #                           size="video_count", color="opportunity_score",
        #                           color_continuous_scale="Viridis",
        #                           hover_name="category",
        #                           hover_data={"video_count":True,"opportunity_score":":.2f"},
        #                           labels={"avg_engagement":"Avg Engagement",
        #                                   "avg_views":"Avg Views",
        #                                   "opportunity_score":"Opportunity"})
        #     fig_opp2.update_layout(height=380)
        #     st.plotly_chart(fig_opp2, use_container_width=True)

    # Thumbnail parallel coordinates — compare all quality attributes at once
    thumb_corr = read_gold("diagnostic/thumbnail_category_correlation")
    if not thumb_corr.empty:
        st.subheader("Thumbnail Quality Profile — Radar Charts")
        st.caption("Visual footprint of high-performing thumbnails. Different categories demand different visual styles.")
        
        # Melt dataframe for polar plotting
        melted_thumb = thumb_corr.melt(
            id_vars=["category", "avg_engagement"], 
            value_vars=["brightness_avg", "contrast_avg", "colorfulness_avg"],
            var_name="Attribute", value_name="Score"
        )
        melted_thumb["Attribute"] = melted_thumb["Attribute"].replace({
            "brightness_avg": "Brightness",
            "contrast_avg": "Contrast",
            "colorfulness_avg": "Colorfulness"
        })

        fig_pc = make_subplots(rows=2, cols=3, specs=[[{'type': 'polar'}]*3]*2, 
                               subplot_titles=GROUPED_CATEGORIES)
        
        for i, cat in enumerate(GROUPED_CATEGORIES):
            row = (i // 3) + 1
            col = (i % 3) + 1
            cat_data = melted_thumb[melted_thumb["category"] == cat]
            if cat_data.empty: continue
            
            # Close the radar loop by repeating the first element
            r_vals = cat_data["Score"].tolist()
            theta_vals = cat_data["Attribute"].tolist()
            r_vals.append(r_vals[0])
            theta_vals.append(theta_vals[0])
            
            fig_pc.add_trace(go.Scatterpolar(
                r=r_vals, theta=theta_vals, fill='toself', name=cat,
                line_color=PALETTE[i % len(PALETTE)], opacity=0.7
            ), row=row, col=col)
            
        fig_pc.update_layout(height=500, showlegend=False)
        fig_pc.update_polars(radialaxis=dict(range=[0, 0.9], showticklabels=False))
        st.plotly_chart(fig_pc, use_container_width=True)

    if not best_m.empty:
        st.subheader("Engagement Trends (Best Time to Post)")
        st.caption("Average engagement rate per category across months. The highest peak represents the best time to post.")
        
        # Calculate full monthly engagement matrix from raw data for smooth curves
        monthly_eng = raw.groupby(["category", "trending_month"], as_index=False)["engagement_rate"].mean()
        import calendar as _cal
        monthly_eng["month_name"] = monthly_eng["trending_month"].apply(lambda m: _cal.month_abbr[int(m)])
        
        fig_bm = px.line(monthly_eng.sort_values(["trending_month", "category"]),
                         x="month_name", y="engagement_rate", color="category",
                         markers=True, line_shape="spline",
                         color_discrete_sequence=PALETTE,
                         labels={"month_name": "Month", "engagement_rate": "Average Engagement Rate", "category": ""})
        
        fig_bm.update_layout(height=450, hovermode="x unified")
        st.plotly_chart(fig_bm, use_container_width=True)

    # if not recs.empty:
    #     st.subheader("Thumbnail Quality Recommendations per Category")
    #     for _, row in recs.iterrows():
    #         with st.expander(f"Quality Metrics: {row['category']}"):
    #             r1, r2, r3 = st.columns(3)
    #             r1.metric("Brightness", row["recommended_brightness"])
    #             r2.metric("Contrast",   row["recommended_contrast"])
    #             r3.metric("Colorfulness", row["recommended_colorfulness"])
    #             st.info(f"**Tip:** {row['tip']}")
    #             # Show correlation image if available
    #             cat_dir = CATEGORY_DIRS.get(row["category"])
    #             if cat_dir and (cat_dir / "thumbnail_engagement_correlation.jpeg").exists():
    #                 st.image(str(cat_dir / "thumbnail_engagement_correlation.jpeg"),
    #                          caption=f"{row['category']} Thumbnail Correlation",
    #                          use_column_width=True)

    # Global thumbnail averages reminder
    # st.subheader("Global Thumbnail Averages Reference")
    # ti1, ti2 = st.columns(2)
    # img1 = NEW_DATA / "Thumbnail_Attributes_Averages.jpeg"
    # img2 = NEW_DATA / "Thumbnail_Engagement_Averages.jpeg"
    # if img1.exists(): ti1.image(str(img1), caption="Attribute Averages", use_column_width=True)
    # if img2.exists(): ti2.image(str(img2), caption="Engagement Averages", use_column_width=True)

st.markdown("---")
# st.caption("Kafka → Bronze → Gold (Delta Lake) · Dashboard by YouTube Trending Analytics Engine")