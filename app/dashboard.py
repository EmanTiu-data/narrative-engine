"""
Narrative Intelligence Engine - Dashboard
Streamlit dashboard for Content Creator Analytics with AI Insights
"""

import streamlit as st
import pandas as pd
import plotly.express as px
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
        self.anomaly = VideoAnomalyDetector(z_threshold=2.0)
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
        generate_key = f"generate_{key_prefix}_{video_id}"
        regenerate_key = f"regenerate_{key_prefix}_{video_id}"
        
        # Show existing insight or generate button
        if existing_insight:
            col1, col2 = st.columns([4, 1])
            
            with col1:
                rating = existing_insight.get("rating", "")
                badge = get_rating_badge(rating)
                st.markdown(f"**💡 INSIGHT** {badge}{rating}")
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
                    channel_avg_for_video = self._calculate_channel_averages(
                        self._get_videos_for_channel(video.get("channel_name", ""))
                    )
                    
                    insight_data = self.insights.generate_video_insight(
                        video_data=video,
                        channel_avg=channel_avg_for_video
                    )
                    
                    self.db.save_video_insight(
                        video_id=video_id,
                        platform="youtube",
                        insight_data=insight_data,
                        channel_name=normalize_name(video.get("channel_name", ""))
                    )
                    
                    st.rerun()
        else:
            if st.button("💡 Generar Insight", key=generate_key):
                insight_data = self.insights.generate_video_insight(
                    video_data=video,
                    channel_avg=channel_avg
                )
                
                self.db.save_video_insight(
                    video_id=video_id,
                    platform="youtube",
                    insight_data=insight_data,
                    channel_name=normalize_name(video.get("channel_name", ""))
                )
                
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
        st.markdown("### AI-Powered Content Creator Analytics")
        
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
        
        # Main content - 3 tabs only
        tab1, tab2, tab3 = st.tabs([
            "📥 Data Collection", 
            "🔍 Topic Analysis", 
            "📊 Analytics & Insights"
        ])
        
        # Data Collection Tab
        with tab1:
            self._render_data_collection(yt_channel, spotify_artist, twitch_channel, 
                                        max_videos, max_comments, collect_btn)
        
        # Topic Analysis Tab
        with tab2:
            self._render_topic_analysis(yt_channel, n_topics)
        
        # Analytics & Insights Tab
        with tab3:
            self._render_analytics(yt_channel, spotify_artist, twitch_channel)
    
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
                        for video in yt_data.get("videos", []):
                            self.db.insert_youtube_video(video)
                        
                        for video in yt_data.get("videos", [])[:max_videos]:
                            comments = self.yt.get_video_comments(video["video_id"], max_comments)
                            self.db.insert_youtube_comments(video["video_id"], comments)
                        
                        st.success(f"✅ YouTube: {len(yt_data.get('videos', []))} videos collected")
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
                        self.db.insert_twitch_stats(twitch_channel, {
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "followers": tw_data.get("total_followers", 0),
                            "followers_gained": 0,
                            "avg_viewers": tw_data.get("avg_viewers_per_video", 0),
                            "stream_hours": len(tw_data.get("videos", [])) * 2,
                            "total_views": tw_data.get("total_views", 0)
                        })
                        
                        followers = tw_data.get("total_followers", 0)
                        followers_display = "N/A (API restricted)" if followers == 0 else followers
                        st.success(f"✅ Twitch: {followers_display}")
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
        
        comments_df = self.db.get_youtube_comments(yt_channel)
        
        if len(comments_df) < 50:
            st.warning(f"Need more comments for LDA. Currently have: {len(comments_df)}")
            st.info("Go to Data Collection tab and collect more data")
            return
        
        self.lda = LDAAnalyzer(n_topics=n_topics)
        comments_text = comments_df["text"].tolist()[:10000]
        result = self.lda.analyze_channel([{"text": c} for c in comments_text])
        
        if "error" in result:
            st.error(result["error"])
            return
        
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
    
    def _render_analytics(self, yt_channel, spotify_artist, twitch_channel):
        """Render analytics and insights tab - unified view"""
        
        st.header("📊 Analytics & Insights")
        
        has_youtube = bool(yt_channel)
        has_spotify = bool(spotify_artist)
        has_twitch = bool(twitch_channel)
        
        if not has_youtube and not has_spotify and not has_twitch:
            st.warning("Enter a YouTube channel, Spotify artist, or Twitch streamer in the sidebar")
            return
        
        # Artist/Streamer Insight Button - UNIFIED
        st.subheader("🎯 Artist/Streamer Insight")
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            generate_artist_insight = st.button("🎯 Generate Insight", type="primary")
        
        with col2:
            if generate_artist_insight:
                with st.spinner("Generating insight..."):
                    if has_youtube:
                        self._render_youtube_artist_insight(yt_channel)
                    if has_spotify:
                        self._render_spotify_artist_insight(spotify_artist)
                    if has_twitch:
                        self._render_twitch_streamer_insight(twitch_channel)
        
        st.divider()
        
        # YouTube Section
        if has_youtube:
            self._render_youtube_top3(yt_channel)
            st.divider()
        
        # Spotify Section
        if has_spotify:
            self._render_spotify_top3(spotify_artist)
            st.divider()
        
        # Twitch Section
        if has_twitch:
            self._render_twitch_top3(twitch_channel, yt_channel)
    
    def _render_youtube_artist_insight(self, yt_channel):
        """Render YouTube artist/streamer insight"""
        videos = self._get_videos_for_channel(yt_channel)
        
        if not videos:
            st.info(f"No YouTube data for {yt_channel}")
            return
        
        channel_insight = self.insights.generate_channel_insight(
            videos=videos,
            channel_name=yt_channel
        )
        
        st.success(f"**📺 YouTube - {yt_channel}**")
        st.write(channel_insight.get("insight_text", ""))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Videos", channel_insight.get("total_videos", 0))
        with col2:
            st.metric("Avg Engagement", f"{channel_insight.get('avg_engagement', 0):.2f}%")
        with col3:
            st.metric("Top Performers", channel_insight.get("high_engagement_videos", 0))
    
    def _render_spotify_artist_insight(self, artist_name):
        """Render Spotify artist insight"""
        # Get tracks for the artist
        tracks_df = self.db.get_spotify_tracks(artist_name)
        
        if tracks_df.empty:
            st.warning(f"⚠️ No hay datos adecuados para '{artist_name}' en Spotify")
            st.info("Spotify no encontró resultados para este artista. Verificá el nombre e intentá de nuevo.")
            return
        
        # Check if data matches the artist name
        spotify_names = tracks_df["artist_name"].unique()
        name_matches = any(artist_name.lower() in name or name in artist_name.lower() for name in spotify_names)
        
        if not name_matches:
            st.warning(f"⚠️ Spotify devolvió '{spotify_names[0] if len(spotify_names) > 0 else 'unknown'}' en lugar de '{artist_name}'")
            st.info("Los resultados pueden no ser exactos. Verificá el nombre del artista.")
        
        # Calculate artist stats
        total_tracks = len(tracks_df)
        total_albums = tracks_df["album_name"].nunique() if "album_name" in tracks_df.columns else 0
        avg_track_position = tracks_df["track_position"].mean() if "track_position" in tracks_df.columns and len(tracks_df) > 0 else 1
        
        st.success(f"**🎵 Spotify - {artist_name}**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tracks", total_tracks)
        with col2:
            st.metric("Albums/Singles", total_albums)
        with col3:
            st.metric("Avg Track Pos", f"{avg_track_position:.1f}")
    
    def _render_youtube_top3(self, yt_channel):
        """Render YouTube top 3 by engagement with insights"""
        
        st.subheader("📺 YouTube Top 3 - Best Performing")
        
        videos = self._get_videos_for_channel(yt_channel)
        
        if not videos:
            st.info("No YouTube videos found. Collect data first.")
            return
        
        # Calculate engagement for each video
        for v in videos:
            views = v.get("views", 0)
            likes = v.get("likes", 0)
            comments = v.get("comments_count", 0)
            v["engagement"] = ((likes + comments * 2) / views * 100) if views > 0 else 0
        
        # Sort by engagement and get top 3
        sorted_videos = sorted(videos, key=lambda x: x.get("engagement", 0), reverse=True)[:3]
        
        # Calculate channel averages
        channel_avg = self._calculate_channel_averages(videos)
        
        for i, video in enumerate(sorted_videos, 1):
            video["channel_name"] = yt_channel
            eng = video.get("engagement", 0)
            
            st.markdown(f"""
            **{i}. {video.get('title', 'Untitled')[:60]}**
            - Engagement: **{eng:.2f}%**
            - Views: {video.get('views', 0):,} | Likes: {video.get('likes', 0):,} | Comments: {video.get('comments_count', 0):,}
            """)
            
            self._render_insight_card(video, channel_avg, f"yt_top_{i}")
            st.divider()
    
    def _render_spotify_top3(self, artist_name):
        """Render Spotify top 3 tracks by album position"""
        
        st.subheader("🎵 Spotify Top 3 - Tracks Principales")
        
        tracks_df = self.db.get_spotify_tracks(artist_name)
        
        if tracks_df.empty:
            return
        
        # Sort by track position (lower position = more important = first track of album)
        if "track_position" in tracks_df.columns:
            sorted_tracks = tracks_df.sort_values("track_position").head(3)
        else:
            sorted_tracks = tracks_df.head(3)
        
        # Calculate average track position for comparison
        avg_position = tracks_df["track_position"].mean() if "track_position" in tracks_df.columns and len(tracks_df) > 0 else 1
        
        for i, (_, track) in enumerate(sorted_tracks.iterrows(), 1):
            track_name = track.get("track_name", "Unknown")
            album_name = track.get("album_name", "Unknown Album")
            track_position = track.get("track_position", 1) if pd.notna(track.get("track_position", 1)) else 1
            
            # Badge based on track position
            if track_position == 1:
                badge = "🔥"
                note = "Single principal del álbum"
            elif track_position <= 3:
                badge = "⭐"
                note = "Track destacado"
            else:
                badge = "📀"
                note = f"Track #{track_position} del álbum"
            
            # Compare to average
            if track_position < avg_position:
                comparison = "🔼 Sobre el promedio del artista"
            elif track_position > avg_position:
                comparison = "🔽 Bajo el promedio del artista"
            else:
                comparison = "➖ En el promedio"
            
            st.markdown(f"""
            **{badge} {i}. {track_name}**
            - Album: {album_name}
            - {note} | {comparison}
            """)
    
    def _get_twitch_channel_data(self, channel_name: str) -> dict:
        """Obtiene datos de Twitch para un canal"""
        try:
            tw_data = self.twitch.collect_channel_data(channel_name)
            return tw_data
        except Exception as e:
            return {"error": str(e)}
    
    def _render_twitch_streamer_insight(self, channel_name: str):
        """Render Twitch streamer insight"""
        tw_data = self._get_twitch_channel_data(channel_name)
        
        if tw_data.get("error"):
            st.warning(f"⚠️ Twitch: {tw_data.get('error')}")
            return
        
        # Calculate stats
        videos = tw_data.get("videos", [])
        total_views = tw_data.get("total_views", 0)
        avg_viewers = tw_data.get("avg_viewers_per_video", 0)
        profile_views = tw_data.get("total_views", 0)  # This is actually profile_views
        is_live = tw_data.get("is_live", False)
        
        st.success(f"**🎮 Twitch - {tw_data.get('display_name', channel_name)}**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Videos", tw_data.get("video_count", 0))
        with col2:
            st.metric("Avg Views/Video", f"{avg_viewers:,.0f}")
        with col3:
            live_status = "🔴 EN VIVO" if is_live else "⚫ Offline"
            st.metric("Status", live_status)
        with col4:
            current_viewers = tw_data.get("current_viewers", 0)
            st.metric("Current Viewers", f"{current_viewers:,}" if is_live else "N/A")
    
    def _render_twitch_top3(self, channel_name: str, yt_channel: str = None):
        """Render Twitch stats with stream recommendation"""
        
        st.subheader("🎮 Twitch Stats & Recommendations")
        
        tw_data = self._get_twitch_channel_data(channel_name)
        
        if tw_data.get("error"):
            st.warning(f"No Twitch data for {channel_name}")
            return
        
        videos = tw_data.get("videos", [])
        
        if not videos:
            st.info("No videos/streams found")
            return
        
        # Calculate stats
        total_streams = len(videos)
        total_views = sum(v.get("view_count", 0) for v in videos)
        avg_views = total_views / total_streams if total_streams > 0 else 0
        
        # Calculate average stream duration
        import re
        total_minutes = 0
        for v in videos:
            duration = v.get("duration", "0m")
            match = re.search(r'(\d+)h|(\d+)m|(\d+)s', duration)
            if match:
                hours = int(match.group(1)) if match.group(1) else 0
                mins = int(match.group(2)) if match.group(2) else 0
                secs = int(match.group(3)) if match.group(3) else 0
                total_minutes += hours * 60 + mins + secs / 60
        
        avg_duration_mins = total_minutes / total_streams if total_streams > 0 else 0
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Streams Analizados", total_streams)
        with col2:
            st.metric("Avg Views/Stream", f"{avg_views:,.0f}")
        with col3:
            st.metric("Avg Duración", f"{avg_duration_mins:.0f} min")
        
        # Generate recommendation based on YouTube data
        recommendation = self._generate_stream_recommendation(tw_data, yt_channel)
        
        st.markdown("---")
        st.markdown(f"### 💡 Recomendación de Streaming")
        st.info(recommendation)
    
    def _generate_stream_recommendation(self, twitch_data: dict, yt_channel: str = None) -> str:
        """Genera recomendación sobre horas de streaming basada en datos de YouTube"""
        
        videos = twitch_data.get("videos", [])
        if not videos:
            return "No hay suficientes datos para generar una recomendación."
        
        # Analyze stream performance
        views_list = [v.get("view_count", 0) for v in videos]
        avg_views = sum(views_list) / len(views_list) if views_list else 0
        
        # Get best performing streams
        best_streams = [v for v in videos if v.get("view_count", 0) > avg_views * 1.5]
        worst_streams = [v for v in videos if v.get("view_count", 0) < avg_views * 0.5]
        
        # Calculate average duration
        import re
        durations = []
        for v in videos:
            duration = v.get("duration", "0m")
            match = re.search(r'(\d+)h|(\d+)m', duration)
            if match:
                hours = int(match.group(1)) if match.group(1) else 0
                mins = int(match.group(2)) if match.group(2) else 0
                durations.append(hours * 60 + mins)
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Build recommendation
        parts = []
        
        # Duration recommendation
        if avg_duration < 120:  # Less than 2 hours
            parts.append(f"⏱️ Tus streams son cortos ({avg_duration:.0f} min avg). Considerá hacer streams más largos (3-4 horas) para builds de audiencia.")
        elif avg_duration > 240:  # More than 4 hours
            parts.append(f"⏱️ Streams largos ({avg_duration:.0f} min avg). Considerá acortar a 2-3 horas para mantener engagement.")
        else:
            parts.append(f"⏱️ Duración promedio buena ({avg_duration:.0f} min). Mantené este ritmo.")
        
        # Frequency recommendation based on performance
        if len(best_streams) > len(videos) * 0.3:
            parts.append(f"📈 Tenés {len(best_streams)} streams con buen rendimiento. Aumentá frecuencia si podés.")
        elif len(worst_streams) > len(videos) * 0.4:
            parts.append(f"📉 Muchos streams con bajo rendimiento. Mejorá el contenido o cambiá horarios.")
        
        # Cross-platform recommendation if YouTube data exists
        if yt_channel:
            yt_videos = self._get_videos_for_channel(yt_channel)
            if yt_videos:
                yt_avg_views = sum(v.get("views", 0) for v in yt_videos) / len(yt_videos)
                twitch_avg_views = avg_views
                
                # Compare performance
                if twitch_avg_views > yt_avg_views * 2:
                    parts.append(f"🎯 Twitch tiene mejor engagement que YouTube ({twitch_avg_views:,.0f} vs {yt_avg_views:,.0f}). Enfocate más en streaming.")
                elif twitch_avg_views < yt_avg_views * 0.5:
                    parts.append(f"🎯 YouTube tiene mejor alcance ({yt_avg_views:,.0f} vs {twitch_avg_views:,.0f}). Considerá priorizar contenido de video.")
                else:
                    parts.append(f"🎯 Ambos plataformas tienen engagement similar. Mantené presencia en ambas.")
        
        return " ".join(parts) if parts else "Continuá con tu estrategia actual. Los datos se ven bien."


def main():
    """Main entry point"""
    dashboard = NarrativeDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
