
# FLOW CHART
```mermaid
flowchart TD
    Col[Data Collection\nCollectorScript.py] --> RawDir[(Raw Files)]
    
    RawDir --> SProd[Search Producer\nSearchKafkaProducer.py]
    RawDir --> TProd[Trending Producer\nTrendingKafkaProducer.py]
    RawDir --> CProd[Comment Producer\nCommentKafkaProducer.py]
    RawDir --> ThProd[Thumbnail Producer\nThumbnailKafkaProducer.py]
    
    SProd --> Kafka[(Kafka Broker)]
    TProd --> Kafka
    CProd --> Kafka
    ThProd --> Kafka
    
    Kafka --> SDelta[Search Delta Processor\nBronze & Silver]
    Kafka --> TDelta[Trending Delta Processor\nBronze & Silver]
    Kafka --> CDelta[Comment Delta Processor\nBronze & Silver]
    Kafka --> ThDelta[Thumbnail Delta Processor\nBronze & Silver]
    
    SDelta --> DeltaDir[(Silver Delta Tables)]
    TDelta --> DeltaDir
    CDelta --> DeltaDir
    ThDelta --> DeltaDir
    
    DeltaDir --> SGold[Search Gold Analysis]
    DeltaDir --> TGold[Trending Gold Analysis]
    
    SGold --> GoldDir[(Gold Delta Tables)]
    TGold --> GoldDir
    
    DeltaDir --> Dash[Streamlit Dashboard\napp.py]
    GoldDir --> Dash
```

### 1. Data Collection (`src/DataCollection`)
The pipeline starts by collecting data from YouTube (comments, search results, trending videos, and thumbnails). `CollectorScript.py` fetches the raw data and saves it to a `data/raw/` directory.

### 2. Stream Processing & Delta Lake (Docker + Kafka + Spark)
All data types follow a **Medallion Architecture** using Spark and Delta Lake, orchestrated through Kafka.
* **Producers**: Specialized producers (`SearchKafkaProducer.py`, `TrendingKafkaProducer.py`, `CommentKafkaProducer.py`, `ThumbnailKafkaProducer.py`) push raw records from disk into dedicated Kafka topics.
* **Bronze Layer**: PySpark processors subscribe to these topics and write raw, immutable records to Delta tables in `data/raw/bronze/`.
* **Silver Layer**: The processors then clean the data, handle schema enforcement, remove duplicates, and enrich the data (e.g., Sentiment Analysis, Language Detection, Image Analytics) before writing to `data/processed/silver/`.
* **Gold Layer**: Aggregated analysis and reporting tables are generated for business-level insights in `data/analysis/gold/`.

### 3. Running the Pipeline
You can run the entire end-to-end pipeline (Collection -> Kafka -> Delta -> Dashboard) using the main entry point:

```bash
python3 -m src.entry
```

Or run individual Docker-based steps:
```bash
docker compose up -d --build
docker compose exec producer python3 -m src.DataCollection.SearchKafkaProducer
docker compose exec producer python3 -m src.DataCollection.CommentKafkaProducer
docker compose exec spark-processor python3 -m src.DataProcessing.CommentProcessorDelta
```

### 4. Interactive Dashboard (`src/Dashboard/app.py`)
The Streamlit application loads the **Silver and Gold Delta Tables** from disk using the high-performance `deltalake` library. It provides a multi-page interactive web UI to visualize metrics like audience sentiment distribution, highest correlating engagement variables, and predictive model benchmarks. 
