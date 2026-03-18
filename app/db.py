"""
SQLite Database Manager for Narrative Intelligence Engine
Handles all data persistence: YouTube, Spotify, Twitch
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd


class DatabaseManager:
    """
    SQLite manager for historical data storage.
    
    Automatically resolves database path relative to project root,
    ensuring consistent behavior across different execution contexts.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.
        
        Args:
            db_path: Optional custom path. If not provided, uses default
                     data/narrative.db relative to project root.
        """
        if db_path is None:
            # Resolve relative to project root
            project_root = Path(__file__).parent.parent
            db_path = str(project_root / "data" / "narrative.db")
        else:
            db_path = str(Path(db_path))
        
        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self._init_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_schema(self):
        """Initialize database schema"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # YouTube videos table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS youtube_videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT UNIQUE NOT NULL,
                channel_name TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                published_at TEXT NOT NULL,
                views INTEGER DEFAULT 0,
                likes INTEGER DEFAULT 0,
                comments_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # YouTube comments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS youtube_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                author TEXT,
                text TEXT NOT NULL,
                like_count INTEGER DEFAULT 0,
                published_at TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES youtube_videos(video_id)
            )
        """)
        
        # YouTube LDA topics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS youtube_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_name TEXT NOT NULL,
                topic_id INTEGER NOT NULL,
                topic_name TEXT,
                topic_words TEXT,
                topic_percentage REAL,
                analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Spotify tracks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spotify_tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id TEXT UNIQUE NOT NULL,
                artist_name TEXT NOT NULL,
                track_name TEXT NOT NULL,
                album_name TEXT,
                release_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Spotify streams table (daily)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spotify_streams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id TEXT NOT NULL,
                stream_date TEXT NOT NULL,
                streams INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (track_id) REFERENCES spotify_tracks(track_id),
                UNIQUE(track_id, stream_date)
            )
        """)
        
        # Twitch stats table (daily)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS twitch_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_name TEXT NOT NULL,
                stat_date TEXT NOT NULL,
                followers INTEGER DEFAULT 0,
                followers_gained INTEGER DEFAULT 0,
                avg_viewers REAL DEFAULT 0,
                stream_hours REAL DEFAULT 0,
                total_views INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(channel_name, stat_date)
            )
        """)
        
        # Correlations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform_a TEXT NOT NULL,
                platform_b TEXT NOT NULL,
                lag_days INTEGER DEFAULT 0,
                pearson_r REAL,
                p_value REAL,
                significance TEXT,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(platform_a, platform_b, lag_days)
            )
        """)
        
        # Anomalies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                entity_name TEXT,
                metric TEXT NOT NULL,
                z_score REAL,
                is_outlier BOOLEAN DEFAULT 1,
                detected_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    # ==================== YouTube ====================
    
    def insert_youtube_video(self, video: Dict) -> int:
        """Insert or update a YouTube video"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO youtube_videos 
            (video_id, channel_name, title, description, published_at, views, likes, comments_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video.get('video_id'),
            video.get('channel_name', ''),
            video.get('title', ''),
            video.get('description', ''),
            video.get('published_at', ''),
            video.get('views', 0),
            video.get('likes', 0),
            video.get('comments_count', 0)
        ))
        
        conn.commit()
        video_id = cursor.lastrowid
        conn.close()
        return video_id
    
    def insert_youtube_comments(self, video_id: str, comments: List[Dict]):
        """Insert YouTube comments"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        for comment in comments:
            cursor.execute("""
                INSERT INTO youtube_comments 
                (video_id, author, text, like_count, published_at, sentiment_label, sentiment_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                video_id,
                comment.get('author', ''),
                comment.get('text', ''),
                comment.get('like_count', 0),
                comment.get('published_at', ''),
                comment.get('sentiment_label'),
                comment.get('sentiment_score')
            ))
        
        conn.commit()
        conn.close()
    
    def get_youtube_comments(self, channel_name: str, limit: int = 10000) -> pd.DataFrame:
        """Get all comments for a channel"""
        conn = self._get_connection()
        query = """
            SELECT yc.*, yv.channel_name
            FROM youtube_comments yc
            JOIN youtube_videos yv ON yc.video_id = yv.video_id
            WHERE yv.channel_name = ?
            ORDER BY yc.published_at DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(channel_name, limit))
        conn.close()
        return df
    
    def insert_topic(self, channel_name: str, topic_id: int, topic_name: str, 
                    topic_words: str, topic_percentage: float):
        """Insert LDA topic"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO youtube_topics 
            (channel_name, topic_id, topic_name, topic_words, topic_percentage)
            VALUES (?, ?, ?, ?, ?)
        """, (channel_name, topic_id, topic_name, topic_words, topic_percentage))
        
        conn.commit()
        conn.close()
    
    def get_topics(self, channel_name: str) -> List[Dict]:
        """Get topics for a channel"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM youtube_topics 
            WHERE channel_name = ?
            ORDER BY topic_percentage DESC
        """, (channel_name,))
        
        topics = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return topics
    
    # ==================== Spotify ====================
    
    def insert_spotify_track(self, track: Dict) -> int:
        """Insert or update a Spotify track"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO spotify_tracks 
            (track_id, artist_name, track_name, album_name, release_date)
            VALUES (?, ?, ?, ?, ?)
        """, (
            track.get('track_id'),
            track.get('artist_name', ''),
            track.get('track_name', ''),
            track.get('album_name', ''),
            track.get('release_date', '')
        ))
        
        conn.commit()
        track_id = cursor.lastrowid
        conn.close()
        return track_id
    
    def insert_spotify_streams(self, track_id: str, streams: List[Dict]):
        """Insert Spotify daily streams"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        for stream in streams:
            cursor.execute("""
                INSERT OR REPLACE INTO spotify_streams 
                (track_id, stream_date, streams)
                VALUES (?, ?, ?)
            """, (
                track_id,
                stream.get('date', ''),
                stream.get('streams', 0)
            ))
        
        conn.commit()
        conn.close()
    
    def get_spotify_streams(self, artist_name: str = None) -> pd.DataFrame:
        """Get Spotify streams"""
        conn = self._get_connection()
        
        if artist_name:
            query = """
                SELECT ss.*, st.artist_name, st.track_name
                FROM spotify_streams ss
                JOIN spotify_tracks st ON ss.track_id = st.track_id
                WHERE st.artist_name = ?
                ORDER BY ss.stream_date
            """
            df = pd.read_sql_query(query, conn, params=(artist_name,))
        else:
            query = """
                SELECT ss.*, st.artist_name, st.track_name
                FROM spotify_streams ss
                JOIN spotify_tracks st ON ss.track_id = st.track_id
                ORDER BY ss.stream_date
            """
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    
    # ==================== Twitch ====================
    
    def insert_twitch_stats(self, channel_name: str, stats: Dict):
        """Insert Twitch daily stats"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO twitch_stats 
            (channel_name, stat_date, followers, followers_gained, avg_viewers, stream_hours, total_views)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            channel_name,
            stats.get('date', datetime.now().strftime('%Y-%m-%d')),
            stats.get('followers', 0),
            stats.get('followers_gained', 0),
            stats.get('avg_viewers', 0),
            stats.get('stream_hours', 0),
            stats.get('total_views', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def get_twitch_stats(self, channel_name: str) -> pd.DataFrame:
        """Get Twitch stats for a channel"""
        conn = self._get_connection()
        query = """
            SELECT * FROM twitch_stats 
            WHERE channel_name = ?
            ORDER BY stat_date
        """
        df = pd.read_sql_query(query, conn, params=(channel_name,))
        conn.close()
        return df
    
    # ==================== Correlations ====================
    
    def insert_correlation(self, platform_a: str, platform_b: str, lag_days: int,
                          pearson_r: float, p_value: float, significance: str):
        """Insert correlation result"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO correlations 
            (platform_a, platform_b, lag_days, pearson_r, p_value, significance)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (platform_a, platform_b, lag_days, pearson_r, p_value, significance))
        
        conn.commit()
        conn.close()
    
    def get_correlations(self) -> pd.DataFrame:
        """Get all correlations"""
        conn = self._get_connection()
        query = "SELECT * FROM correlations ORDER BY ABS(pearson_r) DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    # ==================== Anomalies ====================
    
    def insert_anomaly(self, platform: str, entity_id: str, entity_name: str,
                      metric: str, z_score: float, is_outlier: bool = True):
        """Insert anomaly detection result"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO anomalies 
            (platform, entity_id, entity_name, metric, z_score, is_outlier)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (platform, entity_id, entity_name, metric, z_score, is_outlier))
        
        conn.commit()
        conn.close()
    
    def get_anomalies(self, platform: str = None) -> pd.DataFrame:
        """Get anomalies"""
        conn = self._get_connection()
        
        if platform:
            query = "SELECT * FROM anomalies WHERE platform = ? ORDER BY ABS(z_score) DESC"
            df = pd.read_sql_query(query, conn, params=(platform,))
        else:
            query = "SELECT * FROM anomalies ORDER BY ABS(z_score) DESC"
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    
    # ==================== Utility ====================
    
    def get_channel_stats(self, platform: str, channel_name: str) -> Dict:
        """Get summary stats for a channel"""
        conn = self._get_connection()
        
        if platform == 'youtube':
            query = """
                SELECT 
                    COUNT(*) as video_count,
                    SUM(views) as total_views,
                    SUM(likes) as total_likes,
                    SUM(comments_count) as total_comments
                FROM youtube_videos 
                WHERE channel_name = ?
            """
        elif platform == 'twitch':
            query = """
                SELECT 
                    COUNT(*) as days_tracked,
                    MAX(followers) as total_followers,
                    AVG(avg_viewers) as avg_viewers,
                    SUM(stream_hours) as total_hours
                FROM twitch_stats 
                WHERE channel_name = ?
            """
        else:
            return {}
        
        cursor = conn.cursor()
        cursor.execute(query, (channel_name,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else {}


# Singleton instance
db = DatabaseManager()
