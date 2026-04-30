import os
import time
import logging
import json
import requests
from datetime import datetime

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from pymongo import MongoClient

# -------------------- SETUP --------------------
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise EnvironmentError("YOUTUBE_API_KEY environment variable is required")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:password@mongodb-1:27017,mongodb-2:27017,mongodb-3:27017/?replicaSet=rs0&authSource=admin")
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = mongo_client["youtube_big_data"]
    mongo_client.server_info() # Test connection
    logging.info("Connected to MongoDB Replica Set successfully.")
except Exception as e:
    logging.error(f"Could not connect to MongoDB Replica Set: {e}")
    raise

youtube = build("youtube", "v3", developerKey=API_KEY)

logging.basicConfig(level=logging.INFO)
logging.getLogger("googleapiclient.http").setLevel(logging.ERROR)

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
                
                response = requests.get(img_url, timeout=30)
                # FIX: Corrected status_status to status_code
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    count += 1
            except Exception as e:
                logging.warning(f"Could not download thumbnail for {video_id}: {e}")
                continue
    
    if count > 0:
        logging.info(f"Downloaded {count} new images to {THUMBNAIL_PATH}")

# -------------------- HELPERS --------------------
def get_categories(region="US"):
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
        try:
            request = youtube.search().list(
                part="id",
                q=query,
                type="video",
                maxResults=min(50, max_results - len(video_ids)),
                pageToken=next_page_token
            )
            response = request.execute(num_retries=3) 
            video_ids.extend([item["id"]["videoId"] for item in response["items"]])
            next_page_token = response.get("nextPageToken")
            if not next_page_token: break
        except Exception as e:
            logging.error(f"Search failed for {query}: {e}")
            break 
    return video_ids

def get_trending_videos(region_code="US", max_results=100):
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
    except HttpError as e:
        # Skip videos where comments are disabled without noisy warnings.
        try:
            error_payload = json.loads(e.content.decode("utf-8"))
            reason = error_payload["error"]["errors"][0].get("reason")
            if reason == "commentsDisabled":
                return pd.DataFrame(comments)
        except Exception:
            pass
    except Exception:
        pass
    return pd.DataFrame(comments)

def insert_into_mongodb(df, collection_name):
    """Inserts a Pandas DataFrame into a MongoDB collection."""
    if df is None or df.empty:
        return

    # Replace NaN/NaT with None so MongoDB doesn't throw errors
    clean_df = df.where(pd.notnull(df), None)
    records = clean_df.to_dict(orient="records")
    
    collection = db[collection_name]
    
    try:
        collection.insert_many(records)
        logging.info(f"Inserted {len(records)} fresh records into MongoDB -> '{collection_name}' collection")
    except Exception as e:
        logging.error(f"Failed to insert into MongoDB: {e}")


def append_to_raw_store(df, csv_path, json_path, dedupe_cols=None):
    """Append new rows to existing CSV/JSON raw files."""
    if df is None or df.empty:
        return

    combined = df.copy()

    if os.path.exists(csv_path):
        try:
            existing_csv = pd.read_csv(csv_path)
            combined = pd.concat([existing_csv, combined], ignore_index=True)
        except Exception as e:
            logging.warning(f"Could not read existing CSV {csv_path}: {e}")

    if os.path.exists(json_path):
        try:
            existing_json = pd.read_json(json_path)
            combined = pd.concat([existing_json, combined], ignore_index=True)
        except Exception as e:
            logging.warning(f"Could not read existing JSON {json_path}: {e}")

    if dedupe_cols:
        present_cols = [c for c in dedupe_cols if c in combined.columns]
        if present_cols:
            combined = combined.drop_duplicates(subset=present_cols, keep="last")

    combined = combined.reset_index(drop=True)
    combined.to_csv(csv_path, index=False)
    combined.to_json(json_path, orient="records")

# -------------------- MAIN PIPELINE --------------------
if __name__ == "__main__":
    logging.info("Starting Big Data Collection...")
    category_map = get_categories()

    # 1. TRENDING
    logging.info("Fetching trending videos...")
    trending_ids = get_trending_videos(max_results=100)
    trending_df = get_video_stats(trending_ids, category_map, download_imgs=True)
    trending_df["source"] = "trending"
    append_to_raw_store(
        trending_df,
        os.path.join(DATA_PATH, "trending.csv"),
        os.path.join(DATA_PATH, "trending.json"),
        dedupe_cols=["video_id", "fetched_at"],
    )
    insert_into_mongodb(trending_df, "trending_raw")

    # 2. SEARCH (Unified loop for High Volume)
    search_queries = ["gaming", "news", "music", "tech", "vlogs", "movies", "sports", "machine learning"]
    all_search_stats = []
    all_search_video_ids = set()

    for query in search_queries:
        logging.info(f"Processing query: {query}")
        s_ids = search_videos(query, max_results=100)
        all_search_video_ids.update(s_ids)
        s_df = get_video_stats(s_ids, category_map, download_imgs=True)
        s_df["search_query"] = query
        s_df["source"] = "search"
        
        all_search_stats.append(s_df)
        time.sleep(0.5)

    # Save all combined search results
    search_df = pd.concat(all_search_stats, ignore_index=True)
    append_to_raw_store(
        search_df,
        os.path.join(DATA_PATH, "search.csv"),
        os.path.join(DATA_PATH, "search.json"),
        dedupe_cols=["video_id", "search_query", "fetched_at"],
    )
    insert_into_mongodb(search_df, "search_raw")

    # 3. COMMENTS
    logging.info(f"Collecting comments for videos")
    comments_list = []
    comment_video_ids = list(dict.fromkeys(trending_ids + list(all_search_video_ids)))
    
    for index, vid in enumerate(comment_video_ids):
        if index % 10 == 0:
            logging.info(f"Progress: Fetching comments for video {index + 1} of {len(comment_video_ids)}...")
            
        try:
            c_df = get_video_comments(vid, max_results=50)
            if not c_df.empty: 
                comments_list.append(c_df)
        except Exception as e:
            logging.warning(f"Timeout/Error on video {vid}, skipping...")
            continue
    
    if comments_list:
        comments_df = pd.concat(comments_list, ignore_index=True)
        insert_into_mongodb(comments_df, "comments_raw")

    logging.info("Successfully collected metadata and binary thumbnails.")