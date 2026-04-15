import json
import os
import time
from kafka import KafkaProducer


KAFKA_BOOTSTRAP = "kafka:9092"
KAFKA_TOPIC = "youtube.search.raw"


def get_search_json_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "data", "raw", "search.json")


def load_search_records(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw search JSON not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected search.json to be a JSON array of records.")
    return data


def produce_search_records(records):
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    sent = 0
    for record in records:
        producer.send(KAFKA_TOPIC, value=record)
        sent += 1
        # Small delay helps avoid flooding on low-resource local setups.
        time.sleep(0.01)

    producer.flush()
    producer.close()
    return sent


if __name__ == "__main__":
    path = get_search_json_path()
    print(f"Loading search records from {path} ...")
    search_records = load_search_records(path)
    print(f"Publishing {len(search_records)} records to topic '{KAFKA_TOPIC}' ...")
    sent_count = produce_search_records(search_records)
    print(f"Done. Published {sent_count} search records.")
