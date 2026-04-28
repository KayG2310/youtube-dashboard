import argparse
import logging
import os
import time
from typing import Dict, List, Optional

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "US_Trending.csv")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "US_Trending_Thumbnails.csv")
OUTPUT_COLUMNS = ["video_id", "thumbnail_url"]
THUMBNAIL_QUALITY_ORDER = ["maxres", "standard", "high", "medium", "default"]


def load_youtube_client():
    from dotenv import load_dotenv
    from googleapiclient.discovery import build

    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise EnvironmentError("YOUTUBE_API_KEY environment variable is required")
    return build("youtube", "v3", developerKey=api_key)


def load_video_ids(input_path: str, limit: Optional[int] = None) -> List[str]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Trending CSV not found at: {input_path}")

    df = pd.read_csv(input_path, usecols=["video_id"])
    video_ids = (
        df["video_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    video_ids = [video_id for video_id in dict.fromkeys(video_ids) if video_id]

    if limit is not None:
        video_ids = video_ids[:limit]

    if not video_ids:
        raise ValueError(f"No video_id values found in {input_path}")
    return video_ids


def chunk_list(values: List[str], chunk_size: int) -> List[List[str]]:
    return [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]


def select_thumbnail_url(thumbnails: Dict[str, Dict[str, object]]) -> str:
    for quality in THUMBNAIL_QUALITY_ORDER:
        thumbnail = thumbnails.get(quality)
        if thumbnail and thumbnail.get("url"):
            return str(thumbnail["url"])
    return ""


def get_thumbnail_rows(youtube, video_ids: List[str]) -> List[Dict[str, str]]:
    request = youtube.videos().list(
        part="snippet",
        id=",".join(video_ids),
        maxResults=len(video_ids),
    )
    response = request.execute()

    item_map = {
        item["id"]: item["snippet"].get("thumbnails", {})
        for item in response.get("items", [])
    }

    rows = []
    for video_id in video_ids:
        thumbnails = item_map.get(video_id, {})
        rows.append(
            {
                "video_id": video_id,
                "thumbnail_url": select_thumbnail_url(thumbnails),
            }
        )
    return rows


def create_output_csv(output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_path, index=False, encoding="utf-8-sig")


def append_rows_to_csv(rows: List[Dict[str, str]], output_path: str) -> int:
    if not rows:
        return 0

    output_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_df.to_csv(output_path, mode="a", header=False, index=False, encoding="utf-8-sig")
    return len(output_df)


def collect_thumbnail_urls(input_path: str, output_path: str, limit: Optional[int], sleep: float) -> int:
    youtube = load_youtube_client()
    video_ids = load_video_ids(input_path, limit=limit)
    logging.info("Loaded %d unique video IDs from %s", len(video_ids), input_path)

    create_output_csv(output_path)
    total_rows = 0
    batches = chunk_list(video_ids, 50)

    for index, batch in enumerate(batches, start=1):
        try:
            rows = get_thumbnail_rows(youtube, batch)
        except Exception as e:
            logging.warning("Skipping batch %d/%d: %s", index, len(batches), e)
            rows = [{"video_id": video_id, "thumbnail_url": ""} for video_id in batch]

        written_count = append_rows_to_csv(rows, output_path)
        total_rows += written_count
        logging.info(
            "[%d/%d] appended %d thumbnail rows",
            index,
            len(batches),
            written_count,
        )
        time.sleep(sleep)

    return total_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect YouTube thumbnail URLs for every video_id in US_Trending.csv."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to trending CSV with a video_id column.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path where thumbnail URL CSV will be written.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of videos to process for testing.")
    parser.add_argument("--sleep", type=float, default=0.1, help="Delay between API batches.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    row_count = collect_thumbnail_urls(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        sleep=args.sleep,
    )
    logging.info("Done. Wrote %d rows to %s", row_count, args.output)
