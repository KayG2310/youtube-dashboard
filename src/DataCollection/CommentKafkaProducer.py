import json
import os
import time
from datetime import datetime
from kafka import KafkaProducer


KAFKA_BOOTSTRAP = "kafka:9092"
KAFKA_TOPIC = "youtube.comments.raw"


def get_comments_json_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "data", "raw", "comments.json")


def load_comment_records(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw comments JSON not found at: {path}. Run CollectorScript.py first.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected comments.json to be a JSON array of records.")
    return data


def produce_comment_records(records):
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    sent = 0
    for record in records:
        # Ensure schema compatibility with CommentProcessorDelta
        # CollectorScript provides: video_id, comment_text, like_count, author
        # We add/map fields as needed
        enriched_record = {
            "video_id": record.get("video_id"),
            "comment_id": record.get("comment_id", f"c_{sent}_{int(time.time())}"), # Fallback ID
            "author_name": record.get("author", "Unknown"),
            "comment_text": record.get("comment_text"),
            "like_count": float(record.get("like_count", 0)),
            "published_at": record.get("published_at", datetime.utcnow().isoformat()),
            "fetched_at": record.get("fetched_at", datetime.utcnow().isoformat()),
            "source": record.get("source", "collector"),
        }
        
        producer.send(KAFKA_TOPIC, value=enriched_record)
        sent += 1
        # Small delay helps avoid flooding on low-resource local setups.
        time.sleep(0.01)

    producer.flush()
    producer.close()
    return sent


if __name__ == "__main__":
    try:
        path = get_comments_json_path()
        print(f"Loading comment records from {path} ...")
        comment_records = load_comment_records(path)
        print(f"Publishing {len(comment_records)} records to topic '{KAFKA_TOPIC}' ...")
        sent_count = produce_comment_records(comment_records)
        print(f"Done. Published {sent_count} comment records.")
    except Exception as e:
        print(f"Error: {e}")
