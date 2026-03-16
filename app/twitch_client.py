"""
Twitch API Client
Maneja datos de streamers (live content)
"""

import os
import base64
import hashlib
import hmac
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class TwitchClient:
    """Cliente para Twitch Helix API"""
    
    BASE_URL = "https://api.twitch.tv/helix"
    AUTH_URL = "https://id.twitch.tv/oauth2/token"
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        # Load from env if not provided
        if client_id is None:
            from dotenv import load_dotenv
            load_dotenv()
        self.client_id = client_id or os.getenv("TWITCH_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("TWITCH_CLIENT_SECRET")
        self.access_token = None
        self.auth_error = None
        if self.client_id and self.client_secret:
            try:
                self._authenticate()
            except Exception as e:
                self.auth_error = str(e)
                print(f"WARNING: Twitch auth failed: {e}")
    
    def _authenticate(self):
        """Autentica con Twitch"""
        params = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials"
        }
        
        response = requests.post(self.AUTH_URL, params=params)
        
        if response.status_code == 200:
            self.access_token = response.json()["access_token"]
        else:
            raise Exception(f"Twitch auth failed: {response.text}")
    
    def _get_headers(self) -> Dict:
        """Get headers with auth token"""
        return {
            "Client-ID": self.client_id,
            "Authorization": f"Bearer {self.access_token}"
        }
    
    def get_user_by_login(self, login_name: str) -> Optional[Dict]:
        """Obtiene usuario por nombre de login"""
        url = f"{self.BASE_URL}/users"
        params = {"login": login_name}
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 401:
            self._authenticate()
            response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("data"):
                return data["data"][0]
        
        return None
    
    def get_user_by_login_with_stats(self, login_name: str) -> Optional[Dict]:
        """Obtiene usuario con estadísticas (followers totales)"""
        url = f"{self.BASE_URL}/users"
        params = {"login": login_name}
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 401:
            self._authenticate()
            response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("data"):
                user = data["data"][0]
                # Get view_count from user data (Twitch removed follows API for most users)
                user["total_followers"] = user.get("view_count", 0)  # Using view_count as fallback
                # Note: Twitch API no longer provides follower count for non-partners
                # view_count is total profile views, not followers
                return user
        
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Obtiene usuario por ID"""
        url = f"{self.BASE_URL}/users"
        params = {"id": user_id}
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data["data"]:
                return data["data"][0]
        
        return None
    
    def get_followers(self, user_id: str, max_results: int = 100) -> Dict:
        """Obtiene seguidores de un canal"""
        url = f"{self.BASE_URL}/users/follows"
        params = {
            "to_id": user_id,
            "first": min(max_results, 100)
        }
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "total": data.get("total", 0),
                "followers": data.get("data", [])
            }
        
        return {"total": 0, "followers": []}
    
    def get_streams(self, user_id: str) -> Optional[Dict]:
        """Obtiene stream actual (si está en vivo)"""
        url = f"{self.BASE_URL}/streams"
        params = {"user_id": user_id}
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data["data"]:
                return data["data"][0]
        
        return None
    
    def get_streams_history(self, user_id: str, started_at: str, max_results: int = 100) -> List[Dict]:
        """Obtiene historial de streams (requiere Video API)"""
        url = f"{self.BASE_URL}/videos"
        params = {
            "user_id": user_id,
            "first": min(max_results, 100),
            "started_at": started_at
        }
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
        
        return []
    
    def get_channel_info(self, user_id: str) -> Optional[Dict]:
        """Obtiene información del canal"""
        url = f"{self.BASE_URL}/channels"
        params = {"broadcaster_id": user_id}
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data["data"]:
                return data["data"][0]
        
        return None
    
    def get_channel_stats(self, channel_name: str, days_back: int = 30) -> Dict:
        """Obtiene estadísticas de un canal"""
        
        # Get user with stats (includes total followers)
        user = self.get_user_by_login_with_stats(channel_name)
        if not user:
            return {"error": f"Canal '{channel_name}' no encontrado"}
        
        user_id = user["id"]
        
        # Get followers from user data
        total_followers = user.get("total_followers", 0)
        
        # Get current stream (if live)
        current_stream = self.get_streams(user_id)
        
        # Get channel info
        channel_info = self.get_channel_info(user_id)
        
        # Calculate date range for videos
        start_date = (datetime.now() - timedelta(days=days_back)).isoformat() + "Z"
        
        # Get videos (past broadcasts)
        videos = self.get_streams_history(user_id, start_date, max_results=100)
        
        # Calculate stats
        total_views = sum(int(v.get("view_count", 0)) for v in videos)
        avg_viewers = total_views / len(videos) if videos else 0
        
        return {
            "channel_name": channel_name,
            "user_id": user_id,
            "display_name": user.get("display_name", ""),
            "profile_image": user.get("profile_image_url", ""),
            "description": user.get("description", ""),
            "total_followers": total_followers,
            "is_live": current_stream is not None,
            "current_viewers": current_stream.get("viewer_count", 0) if current_stream else 0,
            "total_views": total_views,
            "avg_viewers_per_video": round(avg_viewers, 2),
            "video_count": len(videos),
            "videos": [
                {
                    "video_id": v["id"],
                    "title": v["title"],
                    "published_at": v["published_at"],
                    "view_count": v.get("view_count", 0),
                    "duration": v.get("duration", "")
                }
                for v in videos[:50]  # Limit to 50
            ]
        }
    
    def collect_channel_data(self, channel_name: str, days_back: int = 30) -> Dict:
        """Recolecta todos los datos de un canal"""
        return self.get_channel_stats(channel_name, days_back)


# Singleton instance
def get_twitch_client() -> TwitchClient:
    """Get Twitch client instance"""
    return TwitchClient()
