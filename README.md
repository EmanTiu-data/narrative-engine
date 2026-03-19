# The Narrative Intelligence Engine

🧠 AI-Powered Content Creator Analytics

## 🎯 Objective

This project was born from the need to understand **why** certain content creators thrive while others don't, by analyzing patterns across multiple platforms (YouTube, Spotify, Twitch).

As a data analyst, the vision is to build an engine that:
1. **Extracts knowledge** from large volumes of comments and metrics
2. **Generates insights** explaining engagement and performance
3. **Detects anomalies** that signal opportunities or potential problems

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/EmanTiu-data/narrative-engine.git
cd narrative-engine

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (automatic, but you can do it manually)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 5. Configure API keys
cp .env.example .env  # If .env.example exists
# Edit .env with your credentials

# 6. Run the dashboard
streamlit run app/dashboard.py
```

---

## 📋 Requirements

- **Python 3.11+**
- YouTube, Spotify, and Twitch APIs (create credentials in their developer portals)

### Required APIs

| Platform | Developer Portal | Notes |
|----------|-----------------|-------|
| YouTube | [Google Cloud Console](https://console.cloud.google.com/) | YouTube Data API v3 |
| Spotify | [Spotify Developer](https://developer.spotify.com/dashboard) | Client Credentials flow |
| Twitch | [Twitch Dev Console](https://dev.twitch.tv/console) | Client Credentials flow |

---

## 📁 Project Structure

```
narrative-engine/
├── app/
│   ├── __init__.py          # Project configuration
│   ├── dashboard.py         # Streamlit interface
│   ├── lda_analyzer.py      # Topic Analysis (NLP)
│   ├── anomaly_detector.py  # Anomaly Detection
│   ├── insights_engine.py   # AI Insights Generator
│   ├── db.py                # Database Manager
│   ├── youtube_client.py    # YouTube API Client
│   ├── spotify_client.py    # Spotify API Client
│   └── twitch_client.py     # Twitch API Client
├── data/                    # Persistent data (auto-created)
│   └── narrative.db         # SQLite database
├── tests/                   # Test suite
│   ├── conftest.py
│   ├── test_*.py
│   ├── edge_cases/
│   └── performance/
├── .github/
│   └── workflows/          # CI/CD pipelines
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

---

## 🔬 Features

### Insights Engine (AI-Powered)

Generates explanations for why content performs well or poorly:

- **Video Insights**: On-demand engagement explanations for individual YouTube videos
- **Channel Insights**: Overall performance analysis for channels/artists/streamers
- **Rating System**: S/A+/A/B/C/D ratings based on engagement metrics

```python
from app.insights_engine import InsightsEngine

engine = InsightsEngine()
insight = engine.generate_video_insight(video_data, channel_avg)
print(insight["insight_text"])
# Output: "🟡A - 151% more engagement than average. 
# Drivers: elevated likes, high comment volume, superior reach."
```

### Topic Analysis (LDA)

Automatically extracts main topics from thousands of comments:

- **Topic Extraction**: Identifies up to N topics using Latent Dirichlet Allocation
- **Automatic Labeling**: Classifies topics into categories (skill, entertainment, frequency, etc.)
- **Example**: `"40% talk about technical skill"` → automatically extracted

```python
from app.lda_analyzer import LDAAnalyzer

analyzer = LDAAnalyzer(n_topics=5)
topics = analyzer.fit_transform(comments_list)
print(topics)
```

### Anomaly Detection

Identifies outliers and unusual patterns:

- **Z-Score**: Detects values beyond 3σ from the mean
- **Isolation Forest**: ML algorithm for complex patterns
- **Combined Detection**: Combines multiple methods for better accuracy

```python
from app.anomaly_detector import VideoAnomalyDetector

detector = VideoAnomalyDetector()
result = detector.analyze_video_metrics(videos)
print(f"Anomalies detected: {result['total_outliers']}")
```

---

## 📊 Dashboard

The dashboard includes 3 main tabs:

### 1. Data Collection
- Collect data from YouTube, Spotify, and Twitch APIs
- Stores videos, tracks, comments, and stats in SQLite

### 2. Topic Analysis (LDA)
- Visualize extracted topics from comments
- Interactive charts showing topic distribution

### 3. Analytics & Insights
- **YouTube Top 3**: Best performing videos with AI-generated insights
- **Spotify Top 3**: Featured tracks with album position comparison
- **Twitch Stats**: Stream metrics with duration recommendations
- **Cross-Platform Comparison**: Recommendations based on all platforms

### Unified Artist/Streamer Insight
Generate a complete analysis with one click:

```
📺 YouTube - Channel Name
- 50 videos analyzed
- Average Engagement: 12.5%
- Top Performers: 8

🎵 Spotify - Artist Name
- 19 tracks across 9 albums
- Average Track Position: 2.3

🎮 Twitch - Streamer Name
- 100 streams analyzed
- Average Views/Stream: 247,898
- Status: 🔴 LIVE
```

---

## 💾 Data Persistence

Data is automatically stored in SQLite:

```python
from app.db import DatabaseManager

db = DatabaseManager()
db.insert_youtube_video(video_data)
db.save_video_insight(video_id, "youtube", insight_data)
db.get_youtube_comments(channel_name)
```

The database is automatically created at `data/narrative.db`.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Tests with coverage
pytest tests/ --cov=app --cov-report=html

# Performance tests only
pytest tests/performance/ -v
```

---

## 🔧 Configuration

### Environment Variables (.env)

```env
# YouTube
YOUTUBE_API_KEY=your_youtube_api_key

# Spotify
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret

# Twitch
TWITCH_CLIENT_ID=your_client_id
TWITCH_CLIENT_SECRET=your_client_secret
```

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| Dashboard | Streamlit | Interactive web interface |
| NLP | scikit-learn, NLTK | Topic extraction and preprocessing |
| ML | scikit-learn | Isolation Forest, clustering |
| Stats | scipy.stats | Pearson correlation |
| Data | pandas, numpy | Data manipulation |
| Database | SQLite | Local persistence |
| Visualization | Plotly | Interactive charts |

---

##  Contributing

1. Fork the repository
2. Create a branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📝 License

MIT License - see LICENSE file for details.

---

## 👤 Author

**EmanTiu** - [@EmanTiu-data](https://github.com/EmanTiu-data)

