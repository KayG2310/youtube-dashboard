import pandas as pd
import os
import re


class SearchDataProcessor:
    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    # -----------------------------
    # Cleaning Functions
    # -----------------------------
    def clean_text(self, text):
        """Remove encoding issues and unwanted characters."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r"[^\x00-\x7F]+", "", text)  # remove weird encoding
        return text.strip()

    def parse_duration(self, duration):
        """Convert ISO duration (PT7M52S) → seconds."""
        if not isinstance(duration, str):
            return 0
        minutes = re.search(r'PT(\d+)M', duration)
        seconds = re.search(r'(\d+)S', duration)

        m = int(minutes.group(1)) if minutes else 0
        s = int(seconds.group(1)) if seconds else 0

        return m * 60 + s

    # -----------------------------
    # Main Processing
    # -----------------------------
    def process(self):
        print(f"Loading raw data from {self.raw_path}...")
        df = pd.read_csv(self.raw_path)

        print("Cleaning and transforming data...")

        # 1. Remove duplicates
        df = df.drop_duplicates(subset=["video_id"])

        # 2. Handle missing values
        df["like_count"] = df["like_count"].fillna(0)
        df["comment_count"] = df["comment_count"].fillna(0)
        df["view_count"] = df["view_count"].fillna(0)

        # 4. Convert timestamps
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce", utc=True)

        # 5. Duration → seconds
        df["duration_sec"] = df["duration"].apply(self.parse_duration)

        # 6. Feature Engineering
        print("Creating features...")

        df["engagement"] = (
            (df["like_count"] + df["comment_count"]) /
            df["view_count"].replace(0, 1)
        )

        df["video_age_days"] = (
            (pd.Timestamp.now(tz="UTC") - df["published_at"]).dt.total_seconds() / 86400
        )

        df["title_length"] = df["title"].apply(lambda x: len(str(x)))

        # 7. Filtering
        df = df[df["view_count"] > 1000]

        # 8. Drop useless columns
        df = df.drop(columns=[
            "duration",
        ], errors="ignore")

        # 9. Save processed data
        print(f"Saving processed data to {self.processed_path}...")
        df.to_csv(self.processed_path, index=False)

        json_path = self.processed_path.replace(".csv", ".json")
        df.to_json(json_path, orient="records", indent=4)

        print("Processing complete!")


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    RAW_FILE = os.path.join(BASE_DIR, "data", "raw", "search.csv")
    PROCESSED_FILE = os.path.join(BASE_DIR, "data", "processed", "search_processed.csv")

    if os.path.exists(RAW_FILE):
        processor = SearchDataProcessor(RAW_FILE, PROCESSED_FILE)
        processor.process()
    else:
        print(f"Raw file not found at {RAW_FILE}")