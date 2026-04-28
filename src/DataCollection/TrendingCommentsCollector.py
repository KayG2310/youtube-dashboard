import argparse
import logging
import os
import time
from typing import Dict, List, Optional

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "US_Trending.csv")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "US_Trending_Comments.csv")
OUTPUT_COLUMNS = ["video_id", "comment_text", "like_count"]


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


def get_top_comments(youtube, video_id: str, max_comments: int = 50) -> List[Dict[str, object]]:
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                order="relevance",
                pageToken=next_page_token,
                textFormat="plainText",
            )
            response = request.execute()
        except Exception as e:
            logging.warning("Skipping %s: %s", video_id, extract_http_error_reason(e))
            break

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(
                {
                    "video_id": video_id,
                    "comment_text": snippet.get("textDisplay", ""),
                    "like_count": int(snippet.get("likeCount", 0)),
                }
            )

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments


def extract_http_error_reason(error: Exception) -> str:
    try:
        return error.error_details[0].get("reason", str(error))
    except Exception:
        return str(error)


def create_output_csv(output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_path, index=False, encoding="utf-8-sig")


def append_comments_to_csv(comments: List[Dict[str, object]], output_path: str) -> int:
    if not comments:
        return 0

    output_df = pd.DataFrame(comments, columns=OUTPUT_COLUMNS)
    output_df.to_csv(output_path, mode="a", header=False, index=False, encoding="utf-8-sig")
    return len(output_df)


def collect_comments(input_path: str, output_path: str, max_comments: int, limit: Optional[int], sleep: float) -> int:
    youtube = load_youtube_client()
    video_ids = load_video_ids(input_path, limit=limit)
    logging.info("Loaded %d unique video IDs from %s", len(video_ids), input_path)

    create_output_csv(output_path)
    total_comments = 0

    for index, video_id in enumerate(video_ids, start=1):
        comments = get_top_comments(youtube, video_id, max_comments=max_comments)
        written_count = append_comments_to_csv(comments, output_path)
        total_comments += written_count
        logging.info(
            "[%d/%d] %s: appended %d comments",
            index,
            len(video_ids),
            video_id,
            written_count,
        )
        time.sleep(sleep)

    return total_comments


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect top YouTube comments for every video_id in US_Trending.csv."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to trending CSV with a video_id column.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path where comments CSV will be written.")
    parser.add_argument("--max-comments", type=int, default=50, help="Maximum comments to collect per video.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of videos to process for testing.")
    parser.add_argument("--sleep", type=float, default=0.1, help="Delay between videos to avoid rapid API calls.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    row_count = collect_comments(
        input_path=args.input,
        output_path=args.output,
        max_comments=args.max_comments,
        limit=args.limit,
        sleep=args.sleep,
    )
    logging.info("Done. Wrote %d rows to %s", row_count, args.output)
