import os
import csv
import requests
from pathlib import Path

def download_thumbnail(video_id, output_dir):
    """Download thumbnail for a given video_id."""
    # Try maxresdefault first, then fallback to hqdefault
    urls = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
    ]
    
    output_path = output_dir / f"{video_id}.jpg"
    
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {video_id}")
                return True
        except Exception as e:
            print(f"Failed to download {video_id} from {url}: {e}")
            continue
    
    print(f"Could not download thumbnail for {video_id}")
    return False

def main():
    # Paths
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "US_Trending_filtered.csv"
    thumbnails_dir = project_root / "data" / "thumbnails_US_Trending"
    
    # Ensure thumbnails directory exists
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV and collect video_ids
    video_ids = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['video_id'].strip()
            if video_id:
                video_ids.add(video_id)
    
    print(f"Found {len(video_ids)} unique video IDs to process.")
    
    # Download thumbnails
    downloaded = 0
    for video_id in video_ids:
        if download_thumbnail(video_id, thumbnails_dir):
            downloaded += 1
    
    print(f"Downloaded {downloaded} thumbnails out of {len(video_ids)}.")

if __name__ == "__main__":
    main()