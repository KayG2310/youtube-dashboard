import os
import sys
from .CommentProcessorDelta import CommentProcessorDelta
from .SearchDataProcessorDelta import SearchDataProcessorDelta
from .ThumbnailDataProcessorDelta import ThumbnailDataProcessorDelta
from .TrendingDataProcessorDelta import TrendingDataProcessorDelta

def main():
    print("--- Starting YouTube Dashboard Delta Lake Pipeline ---")
    
    try:
        print("\n--- Processing YouTube Search Data ---")
        SearchDataProcessorDelta().process()
    except Exception as e:
        print(f"Error processing search data: {e}")

    try:
        print("\n--- Processing YouTube Thumbnail Data ---")
        ThumbnailDataProcessorDelta().process()
    except Exception as e:
        print(f"Error processing thumbnail data: {e}")

    try:
        print("\n--- Processing YouTube Trending Data ---")
        TrendingDataProcessorDelta().process()
    except Exception as e:
        print(f"Error processing trending data: {e}")

    try:
        print("\n--- Processing YouTube Comments Data ---")
        CommentProcessorDelta().process()
    except Exception as e:
        print(f"Error processing comments data: {e}")

    print("\nAll Delta processing tasks finished.")

if __name__ == "__main__":
    main()
