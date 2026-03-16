"""
Narrative Intelligence Engine
The Narrative Intelligence Engine - Advanced NLP, Cross-Platform Correlation & Anomaly Detection
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "narrative.db"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
