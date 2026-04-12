import os
import time
import logging
from datetime import datetime

import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

# -------------------- SETUP --------------------
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise EnvironmentError("YOUTUBE_API_KEY environment variable is required")

youtube = build("youtube", "v3", developerKey=API_KEY)

logging.basicConfig(level=logging.INFO)

# -------------------- PATH FIX --------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")

os.makedirs(DATA_PATH, exist_ok=True)

print("Saving data to:", DATA_PATH)


# -------------------- CATEGORY MAPPING --------------------
def get_categories(region="IN"):
    request = youtube.videoCategories().list(
        part="snippet",
        regionCode=region
    )
    response = request.execute()

    return {item["id"]: item["snippet"]["title"] for item in response["items"]}


# -------------------- VIDEO STATS --------------------
def get_video_stats(video_ids, category_map):
    stats = []

    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids[i:i+50])
        )
        response = request.execute()

        for item in response["items"]:
            stats.append({
                "video_id": item["id"],
                "title": item["snippet"]["title"],
                "channel_title": item["snippet"]["channelTitle"],
                "published_at": item["snippet"]["publishedAt"],
                "category_id": item["snippet"]["categoryId"],
                "category": category_map.get(item["snippet"]["categoryId"], "Unknown"),
                "tags": item["snippet"].get("tags", []),
                "view_count": int(item["statistics"].get("viewCount", 0)),
                "like_count": int(item["statistics"].get("likeCount", 0)),
                "comment_count": int(item["statistics"].get("commentCount", 0)),
                "duration": item["contentDetails"]["duration"],
                "description": item["snippet"]["description"],
                "fetched_at": datetime.utcnow().isoformat()
            })

        time.sleep(0.3)  # faster

    return pd.DataFrame(stats)


# -------------------- SEARCH VIDEOS --------------------
def search_videos(query, max_results=100):
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_results:
        request = youtube.search().list(
            part="id",
            q=query,
            type="video",
            maxResults=min(50, max_results - len(video_ids)),
            pageToken=next_page_token
        )
        response = request.execute()

        video_ids.extend([item["id"]["videoId"] for item in response["items"]])
        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    return video_ids


# -------------------- TRENDING VIDEOS --------------------
def get_trending_videos(region_code="IN", max_results=100):
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_results:
        request = youtube.videos().list(
            part="id",
            chart="mostPopular",
            regionCode=region_code,
            maxResults=min(50, max_results - len(video_ids)),
            pageToken=next_page_token
        )
        response = request.execute()

        video_ids.extend([item["id"] for item in response["items"]])
        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    return video_ids


# -------------------- COMMENTS --------------------
def get_video_comments(video_id, max_results=50):
    comments = []
    next_page_token = None

    try:
        while len(comments) < max_results:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()

            for item in response["items"]:
                c = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "video_id": video_id,
                    "comment_text": c["textDisplay"],
                    "like_count": c["likeCount"],
                    "published_at": c["publishedAt"],
                    "author": c["authorDisplayName"],
                    "fetched_at": datetime.utcnow().isoformat()
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

    except Exception as e:
        logging.warning(f"Comments not available for {video_id}: {e}")

    return pd.DataFrame(comments)


# -------------------- MAIN PIPELINE --------------------
if __name__ == "__main__":

    logging.info("Fetching category mapping...")
    category_map = get_categories()

    # -------- TRENDING --------
    logging.info("Fetching trending videos...")
    trending_ids = get_trending_videos(max_results=100)
    trending_df = get_video_stats(trending_ids, category_map)
    trending_df["source"] = "trending"

    trending_df.to_csv(os.path.join(DATA_PATH, "trending.csv"), index=False)
    trending_df.to_json(os.path.join(DATA_PATH, "trending.json"), orient="records")

    logging.info(f"Collected {len(trending_df)} trending videos")

    # -------- SEARCH --------
    logging.info("Fetching search videos...")
    search_ids = search_videos("machine learning", max_results=100)
    search_df = get_video_stats(search_ids, category_map)
    search_df["source"] = "search"

    search_df.to_csv(os.path.join(DATA_PATH, "search.csv"), index=False)
    search_df.to_json(os.path.join(DATA_PATH, "search.json"), orient="records")

    logging.info(f"Collected {len(search_df)} search videos")

    # -------- COMMENTS --------
    logging.info("Fetching comments...")
    comments_list = []

    for vid in trending_ids[:20]:
        df = get_video_comments(vid, max_results=50)
        comments_list.append(df)

    comments_df = pd.concat(comments_list, ignore_index=True)

    comments_df.to_csv(os.path.join(DATA_PATH, "comments.csv"), index=False)
    comments_df.to_json(os.path.join(DATA_PATH, "comments.json"), orient="records")

    logging.info(f"Collected {len(comments_df)} comments")