# The Narrative Intelligence Engine

🧠 Advanced NLP, Cross-Platform Correlation & Anomaly Detection

## Overview

This project extends Streamer-Pulse with advanced analytics:

- **Fase 1**: NLP with LDA (Topic Extraction)
- **Fase 2**: Cross-Platform Correlation (YouTube + Spotify + Twitch)
- **Fase 3**: Anomaly Detection (Z-Score + Isolation Forest)

## Features

### Fase 1: NLP Avanzado (LDA)
- Topic extraction from 10,000+ comments
- Automatic topic labeling (skill, frequency, entertainment, etc.)
- "40% speak about technical skill" → extracted automatically

### Fase 2: Correlación Multi-Plataforma
- Pearson correlation coefficient
- Lag analysis (does Twitch peak predict YouTube +3 days?)
- Graceful degradation: works even if one platform fails

### Fase 3: Detección de Anomalías
- Z-Score (3σ threshold)
- Isolation Forest for complex patterns
- Automatic outlier exclusion

## Setup

```bash
# Install dependencies
cd narrative-engine
pip install -r requirements.txt

# Run dashboard
streamlit run app/dashboard.py
```

## Configuration

Edit `.env` file:
```
YOUTUBE_API_KEY=your_key
SPOTIFY_CLIENT_ID=your_id
SPOTIFY_CLIENT_SECRET=your_secret
TWITCH_CLIENT_ID=your_id
TWITCH_CLIENT_SECRET=your_secret
```

## Usage

1. Enter channel names in sidebar
2. Click "Collect Data" to fetch from APIs
3. Explore tabs:
   - Data Collection
   - Topic Analysis (LDA)
   - Cross-Platform Correlations
   - Anomaly Detection

## Tech Stack

- Python 3.11+
- Streamlit (dashboard)
- scikit-learn (LDA, Isolation Forest)
- scipy.stats (Pearson correlation)
- SQLite (data persistence)
- NLTK (text preprocessing)
- Plotly (visualization)
