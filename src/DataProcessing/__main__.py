import os
import sys
from .CommentProcessor import CommentProcessor

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Process Comments
    comments_raw = os.path.join(RAW_DIR, "comments.csv")
    comments_processed = os.path.join(PROCESSED_DIR, "comments_processed.csv")

    if os.path.exists(comments_raw):
        print("--- Processing YouTube Comments ---")
        processor = CommentProcessor(comments_raw, comments_processed)
        processor.process()
    else:
        print(f"Warning: Raw comments file not found at {comments_raw}")

    # Future: Add processing for trending.csv and search.csv here
    print("All processing tasks finished.")

if __name__ == "__main__":
    main()
