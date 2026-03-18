"""
Narrative Intelligence Engine - Dashboard
Streamlit dashboard for NLP, Correlation & Anomaly Detection
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from app.db import DatabaseManager, normalize_name
from app.youtube_client import YouTubeClient
from app.spotify_client import SpotifyClient
from app.twitch_client import TwitchClient
from app.lda_analyzer import LDAAnalyzer
from app.correlation import CorrelationEngine, GracefulDegradation
from app.anomaly_detector import VideoAnomalyDetector
from app.insights_engine import InsightsEngine, get_rating_badge


# Page config
st.set_page_config(
    page_title="Narrative Intelligence Engine",
    page_icon="🧠",
    layout="wide"
)


class NarrativeDashboard:
    """Dashboard principal"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.yt = YouTubeClient()
        self.spotify = SpotifyClient()
        self.twitch = TwitchClient()
        self.lda = LDAAnalyzer(n_topics=5, n_top_words=10)
        self.correlation = CorrelationEngine(min_correlation=0.2)  # Lower threshold
        self.graceful = GracefulDegradation()
        self.anomaly = VideoAnomalyDetector(z_threshold=2.0)  # Lower threshold from 3 to 2
        self.insights = InsightsEngine()
    
    def _calculate_channel_averages(self, videos: list) -> dict:
        """Calcula promedios del canal para comparaciones."""
        if not videos:
            return {}
        
        total_views = sum(v.get("views", 0) for v in videos)
        total_likes = sum(v.get("likes", 0) for v in videos)
        total_comments = sum(v.get("comments_count", 0) for v in videos)
        n = len(videos)
        
        avg_views = total_views / n
        avg_likes = total_likes / n
        avg_comments = total_comments / n
        
        # Calculate average engagement
        avg_engagement = 0
        for v in videos:
            views = v.get("views", 0)
            if views > 0:
                likes = v.get("likes", 0)
                comments = v.get("comments_count", 0)
                avg_engagement += ((likes + comments * 2) / views * 100)
        avg_engagement /= n
        
        return {
            "avg_views": avg_views,
            "avg_likes": avg_likes,
            "avg_comments": avg_comments,
            "avg_engagement": avg_engagement
        }
    
    def _render_insight_card(self, video: dict, channel_avg: dict, key_prefix: str = ""):
        """Renderiza una tarjeta de insight para un video."""
        video_id = video.get("video_id", "unknown")
        
        # Check if insight already exists
        existing_insight = self.db.get_video_insight(video_id, "youtube")
        
        # Create unique key for Streamlit widgets
        insight_key = f"insight_{key_prefix}_{video_id}"
        generate_key = f"generate_{key_prefix}_{video_id}"
        regenerate_key = f"regenerate_{key_prefix}_{video_id}"
        
        # Show existing insight or generate button
        if existing_insight:
            col1, col2 = st.columns([4, 1])
            
            with col1:
                rating = existing_insight.get("rating", "")
                badge = get_rating_badge(rating)
                st.markdown(f"**💡 INSIGHT** {badge}{rating} - Guardado: {existing_insight.get('generated_at', '')[:16]}")
                st.write(existing_insight.get("insight_text", ""))
                
                # Show drivers
                drivers = existing_insight.get("drivers", [])
                if drivers:
                    st.write(f"*Drivers: {', '.join(drivers)}*")
                
                # Show tip
                tip = existing_insight.get("tip", "")
                if tip:
                    st.write(f"*Tip: {tip}*")
                
                # Show anomalies
                anomalies = existing_insight.get("anomalies", [])
                for anomaly in anomalies:
                    st.warning(anomaly.get("insight_text", ""))
            
            with col2:
                if st.button("🔄", key=regenerate_key, help="Regenerar insight"):
                    # Delete old insight
                    # Generate new one
                    channel_avg_for_video = self._calculate_channel_averages(
                        self._get_videos_for_channel(video.get("channel_name", ""))
                    )
                    
                    insight_data = self.insights.generate_video_insight(
                        video_data=video,
                        channel_avg=channel_avg_for_video
                    )
                    
                    # Save to DB
                    self.db.save_video_insight(
                        video_id=video_id,
                        platform="youtube",
                        insight_data=insight_data,
                        channel_name=normalize_name(video.get("channel_name", ""))
                    )
                    
                    st.rerun()
        else:
            if st.button("💡 Generar Insight", key=generate_key):
                # Generate insight
                insight_data = self.insights.generate_video_insight(
                    video_data=video,
                    channel_avg=channel_avg
                )
                
                # Save to DB
                self.db.save_video_insight(
                    video_id=video_id,
                    platform="youtube",
                    insight_data=insight_data,
                    channel_name=normalize_name(video.get("channel_name", ""))
                )
                
                # Display
                st.success(f"**Video {get_rating_badge(insight_data['rating'])}{insight_data['rating']}** - {insight_data['insight_text']}")
                
                if insight_data.get("drivers"):
                    st.write(f"*Drivers: {', '.join(insight_data['drivers'])}*")
                
                if insight_data.get("tip"):
                    st.write(f"*Tip: {insight_data['tip']}*")
                
                for anomaly in insight_data.get("anomalies", []):
                    st.warning(anomaly.get("insight_text", ""))
    
    def _get_videos_for_channel(self, channel_name: str) -> list:
        """Obtiene todos los videos de un canal."""
        conn = self.db._get_connection()
        
        query = """
            SELECT DISTINCT 
                yv.video_id,
                yv.title,
                yv.views,
                yv.likes,
                COUNT(yc.id) as comments_count
            FROM youtube_videos yv
            LEFT JOIN youtube_comments yc ON yv.video_id = yc.video_id
            WHERE yv.channel_name = ?
            GROUP BY yv.video_id
        """
        
        videos_df = pd.read_sql_query(query, conn, params=[normalize_name(channel_name)])
        conn.close()
        
        videos = []
        for _, row in videos_df.iterrows():
            videos.append({
                "video_id": row["video_id"],
                "title": row["title"] if pd.notna(row["title"]) else "Untitled",
                "views": row["views"] if pd.notna(row["views"]) else 0,
                "likes": row["likes"] if pd.notna(row["likes"]) else 0,
                "comments_count": row["comments_count"] if pd.notna(row["comments_count"]) else 0
            })
        
        return videos
    
    def run(self):
        """Run dashboard"""
        
        st.title("🧠 The Narrative Intelligence Engine")
        st.markdown("### Advanced NLP, Cross-Platform Correlation & Anomaly Detection")
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            st.subheader("YouTube")
            yt_channel = st.text_input("YouTube Channel", value="werlyb")
            
            st.subheader("Spotify")
            spotify_artist = st.text_input("Spotify Artist", value="")
            
            st.subheader("Twitch")
            twitch_channel = st.text_input("Twitch Channel", value="")
            
            st.subheader("Settings")
            max_videos = st.slider("Max Videos", 10, 50, 50)
            max_comments = st.slider("Max Comments per Video", 50, 500, 100)
            n_topics = st.slider("LDA Topics", 3, 10, 5)
            
            st.markdown("---")
            collect_btn = st.button("📥 Collect Data", type="primary")
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Data Collection", 
            "🔍 Topic Analysis (LDA)", 
            "🔗 Correlations",
            "⚠️ Anomaly Detection"
        ])
        
        # Data Collection Tab
        with tab1:
            self._render_data_collection(yt_channel, spotify_artist, twitch_channel, 
                                        max_videos, max_comments, collect_btn)
        
        # Topic Analysis Tab
        with tab2:
            self._render_topic_analysis(yt_channel, n_topics)
        
        # Correlations Tab
        with tab3:
            self._render_correlations(yt_channel, spotify_artist, twitch_channel)
        
        # Anomaly Detection Tab
        with tab4:
            self._render_anomalies(yt_channel)
    
    def _render_data_collection(self, yt_channel, spotify_artist, twitch_channel,
                               max_videos, max_comments, collect_btn):
        """Render data collection tab"""
        
        st.header("📥 Data Collection")
        
        if collect_btn:
            with st.spinner("Collecting data from platforms..."):
                progress_bar = st.progress(0)
                
                # YouTube
                if yt_channel:
                    progress_bar.progress(25)
                    st.info(f"📺 Collecting YouTube data for {yt_channel}...")
                    
                    yt_data = self.yt.collect_channel_data(yt_channel, max_videos)
                    
                    if "error" not in yt_data:
                        # Save videos
                        for video in yt_data.get("videos", []):
                            self.db.insert_youtube_video(video)
                        
                        # Get comments for each video
                        for video in yt_data.get("videos", [])[:max_videos]:  # Use max_videos
                            comments = self.yt.get_video_comments(video["video_id"], max_comments)
                            self.db.insert_youtube_comments(video["video_id"], comments)
                        
                        st.success(f"✅ YouTube: {len(yt_data.get('videos', []))} videos, comments collected")
                    else:
                        st.error(f"❌ YouTube: {yt_data.get('error')}")
                
                # Spotify
                progress_bar.progress(50)
                if spotify_artist:
                    st.info(f"🎵 Collecting Spotify data for {spotify_artist}...")
                    
                    sp_data = self.spotify.collect_artist_data(spotify_artist)
                    
                    if "error" not in sp_data:
                        for track in sp_data.get("tracks", []):
                            self.db.insert_spotify_track(track)
                        
                        st.success(f"✅ Spotify: {len(sp_data.get('tracks', []))} tracks collected")
                    else:
                        st.warning(f"⚠️ Spotify: {sp_data.get('error')}")
                
                # Twitch
                progress_bar.progress(75)
                if twitch_channel:
                    st.info(f"🎮 Collecting Twitch data for {twitch_channel}...")
                    
                    tw_data = self.twitch.collect_channel_data(twitch_channel)
                    
                    if "error" not in tw_data:
                        # Save current stats
                        self.db.insert_twitch_stats(twitch_channel, {
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "followers": tw_data.get("total_followers", 0),
                            "followers_gained": 0,
                            "avg_viewers": tw_data.get("avg_viewers_per_video", 0),
                            "stream_hours": len(tw_data.get("videos", [])) * 2,  # Estimate
                            "total_views": tw_data.get("total_views", 0)
                        })
                        
                        followers = tw_data.get("total_followers", 0)
                        followers_display = "N/A (API restringida)" if followers == 0 else followers
                        st.success(f"✅ Twitch: {followers_display}, Videos: {tw_data.get('video_count', 0)}, Views: {tw_data.get('total_views', 0):,}")
                    else:
                        st.warning(f"⚠️ Twitch: {tw_data.get('error')}")
                
                progress_bar.progress(100)
        
        # Show collected data summary
        st.subheader("📊 Database Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            yt_comments = self.db.get_youtube_comments(yt_channel)
            st.metric("YouTube Comments", len(yt_comments))
        
        with col2:
            sp_tracks = self.db.get_spotify_tracks()
            st.metric("Spotify Tracks", len(sp_tracks))
        
        with col3:
            if twitch_channel:
                tw_stats = self.db.get_twitch_stats(twitch_channel)
                st.metric("Twitch Records", len(tw_stats))
            else:
                st.metric("Twitch Records", 0)
    
    def _render_topic_analysis(self, yt_channel, n_topics):
        """Render topic analysis tab"""
        
        st.header("🔍 Topic Analysis (LDA)")
        
        if not yt_channel:
            st.warning("Please enter a YouTube channel in the sidebar")
            return
        
        # Get comments
        comments_df = self.db.get_youtube_comments(yt_channel)
        
        if len(comments_df) < 50:
            st.warning(f"Need more comments for LDA. Currently have: {len(comments_df)}")
            st.info("Go to Data Collection tab and collect more data")
            return
        
        # Update LDA topics
        self.lda = LDAAnalyzer(n_topics=n_topics)
        
        # Analyze
        comments_text = comments_df["text"].tolist()[:10000]
        result = self.lda.analyze_channel([{"text": c} for c in comments_text])
        
        if "error" in result:
            st.error(result["error"])
            return
        
        # Display summary
        st.success(f"📊 Analyzed {result['n_comments_analyzed']} comments")
        
        # Topic distribution
        st.subheader("📈 Topic Distribution")
        
        topics_df = pd.DataFrame(result["topics"])
        
        fig = px.bar(
            topics_df, 
            x="topic_name", 
            y="topic_percentage",
            color="topic_name",
            title="Topic Distribution (% of comments)",
            labels={"topic_percentage": "Percentage (%)", "topic_name": "Topic"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Topic details
        st.subheader("📝 Topic Details")
        
        for topic in result["topics"]:
            with st.expander(f"📌 {topic['topic_name']} ({topic['topic_percentage']:.1f}%)"):
                st.write("**Keywords:** " + ", ".join(topic["keywords"]))
                st.write("**All words:** " + ", ".join(topic["topic_words"]))
        
        # Summary
        st.subheader("📋 Summary")
        st.text(result["summary"])
    
    def _render_correlations(self, yt_channel, spotify_artist, twitch_channel):
        """Render correlations tab"""
        
        st.header("🔗 Cross-Platform Correlations")
        
        # Prepare platform data
        platform_data = {}
        
        # YouTube
        if yt_channel:
            yt_data = self.db.get_youtube_comments(yt_channel)
            if len(yt_data) > 0:
                yt_df = yt_data.copy()
                yt_df["date"] = pd.to_datetime(yt_df["published_at"]).dt.date
                yt_agg = yt_df.groupby("date").agg({
                    "text": "count"
                }).reset_index()
                yt_agg.columns = ["date", "comments"]
                platform_data["youtube"] = yt_agg
        
        # Spotify
        if spotify_artist:
            sp_tracks = self.db.get_spotify_tracks(spotify_artist)
            if len(sp_tracks) > 0:
                st.info(f"Spotify: {len(sp_tracks)} tracks collected for {spotify_artist}")
            else:
                st.info(f"Collect Spotify data for {spotify_artist} in Data Collection tab")
        
        # Twitch
        if twitch_channel:
            tw_data = self.db.get_twitch_stats(twitch_channel)
            if len(tw_data) > 0:
                tw_df = tw_data.copy()
                tw_df["date"] = pd.to_datetime(tw_df["stat_date"]).dt.date
                tw_agg = tw_df[["date", "avg_viewers", "followers"]].copy()
                platform_data["twitch"] = tw_agg
        
        if len(platform_data) < 2:
            st.warning("Need data from at least 2 platforms")
            st.info("Collect data first in the Data Collection tab")
            return
        
        # Calculate correlations
        result = self.graceful.calculate_available_correlations(platform_data)
        
        # Show message
        st.info(result.get("message", ""))
        
        # Display significant correlations
        st.subheader("📊 Significant Correlations")
        
        if result.get("significant_correlations"):
            for key, corr in result["significant_correlations"].items():
                r = corr.get("pearson", {}).get("r", 0)
                p = corr.get("pearson", {}).get("p_value", 1)
                lag = corr.get("lag_analysis", {}).get("optimal_lag", 0)
                interp = corr.get("lag_analysis", {}).get("interpretation", "")
                
                st.metric(
                    key.replace("_vs_", " vs "),
                    f"r = {r:.3f}",
                    f"p = {p:.4f}, lag = {lag} days"
                )
                st.caption(interp)
                st.divider()
        else:
            st.info("No significant correlations found yet")
        
        # Show all correlations
        with st.expander("View All Correlations"):
            st.json(result.get("all_correlations", {}))
    
    def _render_anomalies(self, yt_channel):
        """Render anomaly detection tab"""
        
        st.header("⚠️ Anomaly & Engagement Detection")
        
        if not yt_channel:
            st.warning("Please enter a YouTube channel in the sidebar")
            return
        
        # Get videos from database with real metrics
        videos = []
        
        # Try to get real video data from database
        yt_data = self.db.get_youtube_comments(yt_channel)
        
        if len(yt_data) == 0:
            st.warning("No data available")
            st.info("Collect data first in the Data Collection tab")
            return
        
        # Get unique videos with metrics from database (with JOIN to get title)
        conn = self.db._get_connection()
        
        # Normalize channel name (case insensitive)
        yt_channel_normalized = yt_channel.lower().strip()
        
        # Get videos with their titles from youtube_videos table
        query = """
            SELECT DISTINCT 
                yv.video_id,
                yv.title,
                yv.views,
                yv.likes,
                COUNT(yc.id) as comments_count
            FROM youtube_videos yv
            LEFT JOIN youtube_comments yc ON yv.video_id = yc.video_id
            WHERE yv.channel_name = ?
            GROUP BY yv.video_id
            ORDER BY yv.published_at DESC
            LIMIT 50
        """
        
        import pandas as pd
        videos_df = pd.read_sql_query(query, conn, params=[normalize_name(yt_channel)])
        conn.close()
        
        videos = []
        for _, row in videos_df.iterrows():
            videos.append({
                "video_id": row["video_id"],
                "title": row["title"] if pd.notna(row["title"]) else "Untitled",
                "views": row["views"] if pd.notna(row["views"]) else 0,
                "likes": row["likes"] if pd.notna(row["likes"]) else 0,
                "comments_count": row["comments_count"] if pd.notna(row["comments_count"]) else 0
            })
        
        if not videos:
            st.warning("No videos found")
            return
        result = self.anomaly.analyze_video_metrics(videos)
        
        # Show alert
        st.subheader("🚨 Alerts")
        alert_msg = self.anomaly.generate_alert(result)
        if result.get("total_outliers", 0) > 0:
            st.warning(alert_msg)
        else:
            st.success("No anomalies detected")
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Videos Analyzed", result.get("total_videos", 0))
        
        with col2:
            st.metric("Outliers Found", result.get("total_outliers", 0))
        
        with col3:
            st.metric("Outlier %", f"{result.get('outlier_percentage', 0):.1f}%")
        
        # Show outlier videos
        if result.get("outlier_videos"):
            st.subheader("📌 Anomalous Videos")
            
            outlier_df = pd.DataFrame(result["outlier_videos"])
            st.dataframe(outlier_df[["title", "anomalies"]], use_container_width=True)
        
        # Calculate channel averages for comparison
        channel_avg = self._calculate_channel_averages(videos)
        
        # Engagement Ranking Section
        st.divider()
        st.subheader("🏆 Top Engagement Ranking")
        
        # Calculate engagement ranking
        engagement_result = self.anomaly.rank_by_engagement(videos, top_n=5)
        
        if "error" not in engagement_result:
            # Show summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Engagement", f"{engagement_result.get('average_engagement', 0):.2f}%")
            
            with col2:
                threshold = engagement_result.get('interesting_threshold', 0)
                st.metric("'Interesting' Threshold", f">{threshold:.2f}%")
            
            with col3:
                st.metric("Interesting Videos", engagement_result.get('interesting_count', 0))
            
            # Show top 5 engaging videos with insights
            st.subheader("🔥 Top 5 Most Engaging Videos")
            
            for i, video in enumerate(engagement_result.get("top_engaging", [])[:5], 1):
                eng = video.get("engagement_score", 0)
                vs_avg = video.get("engagement_vs_avg_pct", 0)
                interesting = "⭐ INTERESANTE" if video.get("is_interesting") else ""
                
                # Add channel name to video for insights
                video["channel_name"] = yt_channel
                
                st.markdown(f"""
                **{i}. {video.get('title', 'Untitled')[:60]}**
                - Engagement: **{eng:.2f}%** ({vs_avg:+.1f}% vs average) {interesting}
                - Views: {video.get('views', 0):,} | Likes: {video.get('likes', 0):,} | Comments: {video.get('comments_count', 0):,}
                """)
                
                # Render insight card for each video
                self._render_insight_card(video, channel_avg, f"top_{i}")
                
                st.divider()
            
            # Show interesting videos separately
            interesting_videos = engagement_result.get("top_interesting", [])
            if interesting_videos:
                st.subheader("⭐ Videos Interesantes (20%+ above average)")
                interesting_df = pd.DataFrame(interesting_videos)
                st.dataframe(
                    interesting_df[["title", "engagement_score", "engagement_vs_avg_pct", "views", "likes", "comments_count"]],
                    use_container_width=True
                )


def main():
    """Main entry point"""
    dashboard = NarrativeDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
