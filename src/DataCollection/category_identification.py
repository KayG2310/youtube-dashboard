from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('YOUTUBE_API_KEY')
if not API_KEY:
    raise EnvironmentError("YOUTUBE_API_KEY environment variable is required")

youtube = build("youtube", "v3", developerKey=API_KEY)

def get_categories(region="IN"):
    request = youtube.videoCategories().list(
        part="snippet",
        regionCode=region
    )
    response = request.execute()

    categories = {}

    for item in response["items"]:
        categories[item["id"]] = item["snippet"]["title"]

    return categories


if __name__ == "__main__":
    category_map = get_categories()

    for k, v in category_map.items():
        print(k, ":", v)