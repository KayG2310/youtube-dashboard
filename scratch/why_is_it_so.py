import pandas as pd
from pathlib import Path

BASE_DIR = Path("/Users/kamakshigupta/Desktop/Desktop/sem6/youtube-dashboard")
GOLD = BASE_DIR / "data" / "delta_lake" / "gold"

def read_gold(subpath: str) -> pd.DataFrame:
    p = GOLD / subpath
    try:
        import deltalake as dl
        return dl.DeltaTable(str(p)).to_pandas()
    except Exception:
        pq = p / "_data.parquet"
        return pd.read_parquet(str(pq)) if pq.exists() else pd.DataFrame()

cat_vol = read_gold("descriptive/category_volume")
cols = ["category", "avg_views", "avg_likes", "avg_comments", "avg_engagement"]
print(cat_vol[cat_vol["category"].isin(["Music", "Education"])][cols])

# Also check thumbnail attributes
thumb_corr = read_gold("diagnostic/thumbnail_category_correlation")
t_cols = ["category", "brightness_avg", "contrast_avg", "colorfulness_avg", "avg_engagement"]
print("\nThumbnail Correlations:")
print(thumb_corr[thumb_corr["category"].isin(["Music", "Education"])][t_cols])
