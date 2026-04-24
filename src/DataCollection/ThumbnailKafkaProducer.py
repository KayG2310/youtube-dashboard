import json
import os
import time
from datetime import datetime
from collections import Counter
from kafka import KafkaProducer
from PIL import Image
import cv2
import numpy as np


KAFKA_BOOTSTRAP = "kafka:9092"
KAFKA_TOPIC = "youtube.thumbnail.raw"


def get_thumbnail_folder_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "data", "thumbnails")


def analyze_image(image_path):
    # Open image with PIL for basic analysis
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Brightness: average pixel value (0-255)
    brightness = np.mean(img_array)
    
    # Contrast: standard deviation of pixel values
    contrast = np.std(img_array)
    
    # Dominant color: most frequent RGB tuple
    pixels = img_array.reshape(-1, 3)
    pixel_counts = Counter(map(tuple, pixels))
    dominant_color = pixel_counts.most_common(1)[0][0]
    dominant_color_hex = '#%02x%02x%02x' % dominant_color
    
    # Colorfulness: formula from research paper
    rg = img_array[:, :, 0] - img_array[:, :, 1]
    yb = 0.5 * (img_array[:, :, 0] + img_array[:, :, 1]) - img_array[:, :, 2]
    colorfulness = np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    
    # Sharpness: variance of Laplacian (using OpenCV)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        'brightness': float(brightness),
        'contrast': float(contrast),
        'dominant_color': dominant_color_hex,
        'colorfulness': float(colorfulness),
        'sharpness': float(sharpness)
    }


def load_thumbnail_records(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Thumbnail folder not found at: {folder_path}")
    
    records = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    print(f"Found {len(image_files)} image files to analyze.")
    
    for i, filename in enumerate(image_files):
        print(f"Analyzing image {i+1}/{len(image_files)}: {filename}")
        image_path = os.path.join(folder_path, filename)
        video_id = os.path.splitext(filename)[0]  # Assume filename is video_id
        
        try:
            analysis = analyze_image(image_path)
            record = {
                'video_id': video_id,
                'thumbnail_url': image_path,
                'fetched_at': datetime.utcnow().isoformat() + 'Z',
                'source': 'local_analysis',
                **analysis
            }
            records.append(record)
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
    
    if not records:
        raise ValueError("No valid thumbnail images found in the folder.")
    return records


def produce_thumbnail_records(records):
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
    folder_path = get_thumbnail_folder_path()
    print(f"Analyzing thumbnail images from {folder_path} ...")
    thumbnail_records = load_thumbnail_records(folder_path)
    print(f"Publishing {len(thumbnail_records)} records to topic '{KAFKA_TOPIC}' ...")
    sent_count = produce_thumbnail_records(thumbnail_records)
    print(f"Done. Published {sent_count} thumbnail records.")