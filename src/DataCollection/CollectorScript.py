import os
import time
import logging
import requests
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

# -------------------- PATHS --------------------
# Maps to /app/data inside the container
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
THUMBNAIL_PATH = os.path.join(BASE_DIR, "data", "thumbnails")

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(THUMBNAIL_PATH, exist_ok=True)

# -------------------- THUMBNAIL DOWNLOADER --------------------
def download_thumbnails(video_items):
    """
    Downloads high-res thumbnails to create a 'Big Data' binary dataset.
    """
    count = 0
    for item in video_items:
        video_id = item["id"]
        thumbnails = item["snippet"].get("thumbnails", {})
        # Quality fallback: maxres (best for volume) -> high -> standard
        img_info = thumbnails.get("maxres") or thumbnails.get("high") or thumbnails.get("standard")
        
        if img_info:
            img_url = img_info["url"]
            try:
                # FIX: Check if file already exists to save time/bandwidth
                file_path = os.path.join(THUMBNAIL_PATH, f"{video_id}.jpg")
                if os.path.exists(file_path):
                    continue

                response = requests.get(img_url, timeout=10)
                # FIX: Corrected status_status to status_code
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    count += 1
            except Exception as e:
                logging.warning(f"Could not download thumbnail for {video_id}: {e}")
    
    if count > 0:
        logging.info(f"Downloaded {count} new images to {THUMBNAIL_PATH}")

# -------------------- HELPERS --------------------
def get_categories(region="IN"):
    request = youtube.videoCategories().list(part="snippet", regionCode=region)
    response = request.execute()
    return {item["id"]: item["snippet"]["title"] for item in response["items"]}

def get_video_stats(video_ids, category_map, download_imgs=True):
    stats = []
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids[i:i+50])
        )
        response = request.execute()
        items = response["items"]

        if download_imgs:
            download_thumbnails(items)

        for item in items:
            stats.append({
                "video_id": item["id"],
                "title": item["snippet"]["title"],
                "channel_title": item["snippet"]["channelTitle"],
                "published_at": item["snippet"]["publishedAt"],
                "category_id": item["snippet"]["categoryId"],
                "category": category_map.get(item["snippet"]["categoryId"], "Unknown"),
                "view_count": int(item["statistics"].get("viewCount", 0)),
                "like_count": int(item["statistics"].get("likeCount", 0)),
                "comment_count": int(item["statistics"].get("commentCount", 0)),
                "duration": item["contentDetails"]["duration"],
                "thumbnail_url": item["snippet"]["thumbnails"].get("maxres", {}).get("url", "N/A"),
                "fetched_at": datetime.utcnow().isoformat()
            })
        time.sleep(0.3)
    return pd.DataFrame(stats)

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
        if not next_page_token: break
    return video_ids

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
        if not next_page_token: break
    return video_ids

def get_video_comments(video_id, max_results=50):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=max_results, textFormat="plainText"
        )
        response = request.execute()
        for item in response["items"]:
            c = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "comment_text": c["textDisplay"],
                "like_count": c["likeCount"],
                "author": c["authorDisplayName"]
            })
    except Exception: pass
    return pd.DataFrame(comments)

# -------------------- MAIN PIPELINE --------------------
if __name__ == "__main__":
    logging.info("Starting Big Data Collection...")
    category_map = get_categories()

    # 1. TRENDING
    logging.info("Fetching trending videos...")
    trending_ids = get_trending_videos(max_results=100)
    trending_df = get_video_stats(trending_ids, category_map, download_imgs=True)
    trending_df["source"] = "trending"
    trending_df.to_json(os.path.join(DATA_PATH, "trending.json"), orient="records")

    # 2. SEARCH (Unified loop for High Volume)
    search_queries = ["gaming", "news", "music", "tech", "vlogs", "movies", "sports", "machine learning"]
    all_search_stats = []

    for query in search_queries:
        logging.info(f"Processing query: {query}")
        s_ids = search_videos(query, max_results=100)
        s_df = get_video_stats(s_ids, category_map, download_imgs=True)
        s_df["search_query"] = query
        s_df["source"] = "search"
        
        # Backward compatibility for your original search.json analysis
        if query == "machine learning":
            s_df.to_json(os.path.join(DATA_PATH, "search.json"), orient="records")
            
        all_search_stats.append(s_df)
        time.sleep(0.5)

    # Master file for Spark Multimodal Analysis
    final_search_df = pd.concat(all_search_stats, ignore_index=True)
    final_search_df.to_json(os.path.join(DATA_PATH, "search_master.json"), orient="records")

    # 3. COMMENTS
    logging.info("Collecting comments...")
    comments_list = []
    for vid in trending_ids[:20]:
        c_df = get_video_comments(vid, max_results=50)
        if not c_df.empty: comments_list.append(c_df)
    
    if comments_list:
        pd.concat(comments_list, ignore_index=True).to_json(
            os.path.join(DATA_PATH, "comments.json"), orient="records"
        )

    logging.info("Successfully collected metadata and binary thumbnails.")