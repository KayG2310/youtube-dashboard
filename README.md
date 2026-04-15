
# FLOW CHART
```mermaid
flowchart TD
    Col[Data Collection\nCollectorScript.py] --> RawDir[(Raw Files)]
    
    RawDir --> CmtProc[Comment Processor\nClean, Sentiment, LangDetect]
    RawDir --> TrdProc[Trending Processor\nSummarize Trending Data]
    RawDir --> TmbProc[Thumbnail Processor\nImage Analytics]
    
    CmtProc --> ProcDir[(Processed CSV & JSON)]
    TrdProc --> ProcDir
    TmbProc --> ProcDir
    
    RawDir --> Prod[Kafka Producer\nSearchKafkaProducer.py]
    Prod --> Kafka[(Kafka Broker in Docker)]
    Kafka --> DeltaProc[Spark Delta Processor\nBronze & Silver Tables]
    DeltaProc --> DeltaDir[(Silver Delta Tables)]
    DeltaDir --> DeltaGold[Spark Gold Analysis\nsearch_analysis_delta.py]
    DeltaGold --> DeltaGoldDir[(Gold Delta Tables)]
    
    ProcDir --> Dash[Streamlit Dashboard\napp.py]
    DeltaDir --> Dash
    DeltaGoldDir --> Dash
```
### 1. Data Collection (`src/DataCollection`)
The pipeline starts by collecting data from YouTube (comments, search results, trending videos, and thumbnails). `CollectorScript.py` fetches the raw data and saves it to a `data/raw/` directory.

### 2. Batch Processing (`src/DataProcessing`)
Once data is collected, several specialized, offline Python scripts run over the raw data to clean it and build analytical features:
* **`CommentProcessor.py`**: Scrubs out URLs/special characters, runs Sentiment Analysis using `TextBlob`, detects comment languages using `langdetect`, and generates engagement metrics. 
* **`trending_processing.py` & `Thumbnail_processing.py`**: Similarly aggregate processing metrics for the trending and image data.
This clean output is saved as CSV and JSON files in `data/processed/`.

### 3. Stream Processing & Delta Lake (Docker + Kafka + Spark)
For search records, the pipeline simulates a real-time data streaming architecture. You have a `docker-compose.yml` file standing up Zookeeper and a Kafka message broker.
* **Producer (`SearchKafkaProducer.py`)**: Pushes extracted raw search records into Kafka to simulate a real-time stream.
* **PySpark Processor (`SearchDataProcessorDelta.py`)**: Subscribes to the Kafka stream and processes the data using a "Medallion Architecture". It drops unstructured data into **Bronze Delta Tables**, and transforms/cleans it into **Silver Delta Tables**.
* **Gold Analytics (`search_analysis_delta.py`)**: Once Silver data is ready, Spark generates your **Gold Delta Tables**. These Tables contain heavily aggregated domain-level insights (e.g., query performance, forecasting predictions, channel leaderboards, etc).

### 4. Interactive Dashboard (`src/Dashboard/app.py`)
Finally, all this processed structure is brought together. A Streamlit application loads the processed CSVs, JSON, and Delta Lake Tables from disk. It creates a multi-page interactive web UI relying on `plotly` to visualize metrics like audience sentiment distribution, highest correlating engagement variables, and predictive model benchmarks. 

Everything is wired to run sequentially via the main entry point pipeline script at `src/entry/__main__.py`, which executes all processing scripts from top-to-bottom and finally opens the actual dashboard view. 