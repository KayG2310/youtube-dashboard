import pandas as pd
import os
import re
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from datetime import datetime

# To ensure consistent results for language detection
DetectorFactory.seed = 0

class CommentProcessor:
    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    def clean_text(self, text):
        """Removes URLs, special characters and tidies up text."""
        if not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters but keep punctuation and emojis for now
        text = text.strip()
        return text

    def get_sentiment(self, text):
        """Returns sentiment polarity score."""
        if not text:
            return 0.0
        try:
            return TextBlob(text).sentiment.polarity
        except:
            return 0.0

    def detect_language(self, text):
        """Detects the language of the comment."""
        if not text or len(text) < 3:
            return "unknown"
        try:
            return detect(text)
        except:
            return "unknown"

    def process(self):
        print(f"Loading raw comments from {self.raw_path}...")
        df = pd.read_csv(self.raw_path)

        print("Cleaning data and extracting features...")
        # 1. Cleaning
        df['clean_text'] = df['comment_text'].apply(self.clean_text)
        
        # 2. Basic Metrics
        df['char_count'] = df['clean_text'].apply(len)
        df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
        
        # 3. Sentiment Analysis
        print("Performing sentiment analysis...")
        df['sentiment_score'] = df['clean_text'].apply(self.get_sentiment)
        df['sentiment_label'] = df['sentiment_score'].apply(
            lambda s: 'positive' if s > 0.1 else ('negative' if s < -0.1 else 'neutral')
        )
        
        # 4. Language Detection
        print("Detecting languages...")
        df['language'] = df['clean_text'].apply(self.detect_language)
        
        # 5. Engagement Metrics
        # Simple engagement score: like_count weighted by normalized character count
        max_likes = df['like_count'].max() if df['like_count'].max() > 0 else 1
        df['engagement_score'] = (df['like_count'] / max_likes) * 100

        # Save processed data
        print(f"Saving processed data to {self.processed_path}...")
        df.to_csv(self.processed_path, index=False)
        
        # Also save as JSON for easy dashboard integration
        json_path = self.processed_path.replace('.csv', '.json')
        df.to_json(json_path, orient='records', indent=4)
        
        print("Processing complete!")

if __name__ == "__main__":
    # Internal test/example run
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_FILE = os.path.join(BASE_DIR, "data", "raw", "comments.csv")
    PROCESSED_FILE = os.path.join(BASE_DIR, "data", "processed", "comments_processed.csv")
    
    if os.path.exists(RAW_FILE):
        processor = CommentProcessor(RAW_FILE, PROCESSED_FILE)
        processor.process()
    else:
        print(f"Raw file not found at {RAW_FILE}")
