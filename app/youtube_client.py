"""
YouTube API Client
Reutilizado y extendido de Streamer-Pulse
"""

import os
import requests
from typing import List, Dict, Optional
from datetime import datetime


class YouTubeClient:
    """Cliente para YouTube Data API v3"""
    
    BASE_URL = "https://www.googleapis.com/youtube/v3"
    
    def __init__(self, api_key: str = None):
        # Load from env if not provided
        if api_key is None:
            from dotenv import load_dotenv
            load_dotenv()
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
    
    def get_channel_id(self, channel_name: str) -> Optional[str]:
        """Obtiene el ID del canal por nombre"""
        url = f"{self.BASE_URL}/channels"
        params = {
            "part": "id,snippet,statistics",
            "forHandle": f"@{channel_name}",
            "key": self.api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if "items" in data and data["items"]:
            return data["items"][0]["id"]
        
        # Fallback: buscar por nombre
        params = {
            "part": "id,snippet,statistics",
            "q": channel_name,
            "type": "channel",
            "key": self.api_key
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        if "items" in data and data["items"]:
            return data["items"][0]["id"]
        
        return None
    
    def get_channel_stats(self, channel_id: str) -> Dict:
        """Obtiene estadísticas del canal"""
        url = f"{self.BASE_URL}/channels"
        params = {
            "part": "statistics,snippet",
            "id": channel_id,
            "key": self.api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if "items" in data and data["items"]:
            item = data["items"][0]
            return {
                "channel_id": channel_id,
                "title": item["snippet"]["title"],
                "subscriber_count": int(item["statistics"].get("subscriberCount", 0)),
                "view_count": int(item["statistics"].get("viewCount", 0)),
                "video_count": int(item["statistics"].get("videoCount", 0)),
            }
        
        return {}
    
    def get_channel_videos(self, channel_id: str, max_results: int = 50) -> List[Dict]:
        """Obtiene videos del canal usando el playlist de uploads"""
        
        # First get the channel's uploads playlist ID
        channel_url = f"{self.BASE_URL}/channels"
        channel_params = {
            "part": "contentDetails",
            "id": channel_id,
            "key": self.api_key
        }
        
        response = requests.get(channel_url, params=channel_params)
        data = response.json()
        
        if "items" not in data or not data["items"]:
            return []
        
        uploads_playlist_id = data["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        
        # Now get videos from the uploads playlist
        playlist_url = f"{self.BASE_URL}/playlistItems"
        playlist_params = {
            "part": "snippet",
            "playlistId": uploads_playlist_id,
            "maxResults": min(max_results, 50),
            "key": self.api_key
        }
        
        videos = []
        
        while len(videos) < max_results:
            response = requests.get(playlist_url, params=playlist_params)
            data = response.json()
            
            if "items" not in data:
                break
            
            for item in data["items"]:
                snippet = item["snippet"]
                videos.append({
                    "video_id": snippet["resourceId"]["videoId"],
                    "title": snippet["title"],
                    "published_at": snippet["publishedAt"],
                    "description": snippet.get("description", "")
                })
            
            # Check for next page
            if "nextPageToken" in data:
                playlist_params["pageToken"] = data["nextPageToken"]
            else:
                break
        
        return videos[:max_results]
    
    def get_video_details(self, video_ids: List[str]) -> List[Dict]:
        """Obtiene detalles de múltiples videos (views, likes, comments)"""
        if not video_ids:
            return []
        
        url = f"{self.BASE_URL}/videos"
        params = {
            "part": "statistics,snippet",
            "id": ",".join(video_ids),
            "key": self.api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        videos = []
        if "items" in data:
            for item in data["items"]:
                stats = item.get("statistics", {})
                videos.append({
                    "video_id": item["id"],
                    "title": item["snippet"]["title"],
                    "published_at": item["snippet"]["publishedAt"],
                    "views": int(stats.get("viewCount", 0)),
                    "likes": int(stats.get("likeCount", 0)),
                    "comments": int(stats.get("commentCount", 0)),
                    "description": item["snippet"].get("description", "")
                })
        
        return videos
    
    def get_video_comments(self, video_id: str, max_comments: int = 100) -> List[Dict]:
        """Obtiene comentarios de un video"""
        url = f"{self.BASE_URL}/commentThreads"
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(max_comments, 100),
            "order": "time",
            "key": self.api_key
        }
        
        comments = []
        
        while len(comments) < max_comments:
            response = requests.get(url, params=params)
            data = response.json()
            
            if "items" not in data:
                break
            
            for item in data["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "author": comment["authorDisplayName"],
                    "text": comment["textDisplay"],
                    "like_count": comment["likeCount"],
                    "published_at": comment["publishedAt"]
                })
                
                if len(comments) >= max_comments:
                    break
            
            if "nextPageToken" not in data:
                break
            
            params["pageToken"] = data["nextPageToken"]
        
        return comments
    
    def collect_channel_data(self, channel_name: str, max_videos: int = 50, 
                            max_comments_per_video: int = 100) -> Dict:
        """Recolecta todos los datos de un canal"""
        
        # Obtener channel ID
        channel_id = self.get_channel_id(channel_name)
        if not channel_id:
            return {"error": f"Canal '{channel_name}' no encontrado"}
        
        # Obtener stats del canal
        channel_stats = self.get_channel_stats(channel_id)
        
        # Obtener videos
        videos = self.get_channel_videos(channel_id, max_videos)
        
        # Obtener detalles de videos
        video_ids = [v["video_id"] for v in videos]
        video_details = self.get_video_details(video_ids)
        
        # Combinar datos
        enriched_videos = []
        for video in videos:
            video_id = video["video_id"]
            details = next((v for v in video_details if v["video_id"] == video_id), {})
            
            enriched_videos.append({
                "video_id": video_id,
                "channel_name": channel_name,
                "title": video.get("title", ""),
                "description": video.get("description", ""),
                "published_at": video.get("published_at", ""),
                "views": details.get("views", 0),
                "likes": details.get("likes", 0),
                "comments_count": details.get("comments", 0),
            })
        
        return {
            "channel_name": channel_name,
            "channel_id": channel_id,
            "channel_stats": channel_stats,
            "videos": enriched_videos
        }
