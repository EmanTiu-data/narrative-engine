"""
Spotify API Client
Maneja datos de canciones/podcasts (on-demand, no live)
"""

import os
import base64
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class SpotifyClient:
    """Cliente para Spotify API (Tracks & Podcasts - On Demand)"""
    
    BASE_URL = "https://api.spotify.com/v1"
    AUTH_URL = "https://accounts.spotify.com/api/token"
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        # Load from env if not provided
        if client_id is None:
            from dotenv import load_dotenv
            load_dotenv()
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
        self.access_token = None
        self.auth_error = None
        if self.client_id and self.client_secret:
            try:
                self._authenticate()
            except Exception as e:
                self.auth_error = str(e)
                print(f"WARNING: Spotify auth failed: {e}")
    
    def _get_auth_token(self) -> str:
        """Genera token de autenticación"""
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_string.encode('utf-8')
        auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')
        
        headers = {
            "Authorization": f"Basic {auth_base64}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {"grant_type": "client_credentials"}
        
        response = requests.post(self.AUTH_URL, headers=headers, data=data)
        
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            raise Exception(f"Spotify auth failed: {response.text}")
    
    def _authenticate(self):
        """Autentica con Spotify"""
        self.access_token = self._get_auth_token()
    
    def _get_headers(self) -> Dict:
        """Get headers with auth token"""
        return {
            "Authorization": f"Bearer {self.access_token}"
        }
    
    def search_artist(self, artist_name: str) -> Optional[Dict]:
        """Busca un artista"""
        url = f"{self.BASE_URL}/search"
        params = {
            "q": artist_name,
            "type": "artist",
            "limit": 1
        }
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 401:
            self._authenticate()
            response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data["artists"]["items"]:
                return data["artists"]["items"][0]
        
        return None
    
    def get_artist_top_tracks(self, artist_id: str, market: str = "US") -> List[Dict]:
        """Obtiene los top tracks de un artista"""
        url = f"{self.BASE_URL}/artists/{artist_id}/top-tracks"
        params = {"market": market}
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 200:
            data = response.json()
            return [
                {
                    "track_id": track["id"],
                    "track_name": track["name"],
                    "album_name": track["album"]["name"],
                    "album_image": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
                    "popularity": track["popularity"],
                    "duration_ms": track["duration_ms"],
                    "preview_url": track.get("preview_url")
                }
                for track in data.get("tracks", [])
            ]
        
        return []
    
    def get_album_tracks(self, album_id: str) -> List[Dict]:
        """Obtiene los tracks de un álbum"""
        url = f"{self.BASE_URL}/albums/{album_id}/tracks"
        params = {"limit": 50}
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 200:
            data = response.json()
            return [
                {
                    "track_id": track["id"],
                    "track_name": track["name"],
                    "track_number": track.get("track_number", 0),
                    "duration_ms": track.get("duration_ms", 0)
                }
                for track in data.get("items", [])
            ]
        
        return []
    
    def get_artist_albums(self, artist_id: str, limit: int = 20) -> List[Dict]:
        """Obtiene los álbumes de un artista"""
        url = f"{self.BASE_URL}/artists/{artist_id}/albums"
        params = {"limit": limit, "include_groups": "album,single"}
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 200:
            data = response.json()
            return [
                {
                    "album_id": album["id"],
                    "album_name": album["name"],
                    "release_date": album["release_date"],
                    "album_type": album["album_type"],
                    "total_tracks": album["total_tracks"],
                    "image_url": album["images"][0]["url"] if album["images"] else None
                }
                for album in data["items"]
            ]
        
        return []
    
    def get_track(self, track_id: str) -> Optional[Dict]:
        """Obtiene detalles de una canción"""
        url = f"{self.BASE_URL}/tracks/{track_id}"
        
        response = requests.get(url, headers=self._get_headers())
        
        if response.status_code == 200:
            track = response.json()
            return {
                "track_id": track["id"],
                "track_name": track["name"],
                "artist_name": track["artists"][0]["name"],
                "album_name": track["album"]["name"],
                "release_date": track["album"]["release_date"],
                "popularity": track["popularity"],
                "duration_ms": track["duration_ms"]
            }
        
        return None
    
    def get_artist_stats(self, artist_name: str) -> Dict:
        """Obtiene estadísticas completas de un artista"""
        artist = self.search_artist(artist_name)
        
        if not artist:
            return {"error": f"Artista '{artist_name}' no encontrado"}
        
        # Check if the artist name matches reasonably
        returned_name = artist.get("name", "").lower()
        search_name = artist_name.lower()
        
        # Simple check - if names are very different, flag it
        name_match_threshold = 0.5
        if returned_name and search_name:
            # Check if search name is in returned name or vice versa
            matches = search_name in returned_name or returned_name in search_name
            if not matches and len(search_name) > 3:
                # Names don't match well, but still return data with warning
                artist["_name_warning"] = f"Buscaste '{artist_name}' pero Spotify devolvió '{artist.get('name')}'"
        
        artist_id = artist["id"]
        
        # Get top tracks
        top_tracks = self.get_artist_top_tracks(artist_id)
        
        # Get albums
        albums = self.get_artist_albums(artist_id, limit=10)
        
        return {
            "artist_name": artist.get("name", artist_name),
            "artist_id": artist_id,
            "followers": artist.get("followers", {}).get("total", 0),
            "popularity": artist.get("popularity", 0),
            "genres": artist.get("genres", []),
            "top_tracks": top_tracks[:10],
            "recent_albums": albums[:10],
            "image_url": artist["images"][0]["url"] if artist.get("images") else None,
            "_name_warning": artist.get("_name_warning")
        }
    
    def search_tracks(self, artist_name: str, limit: int = 50) -> List[Dict]:
        """Busca tracks de un artista"""
        url = f"{self.BASE_URL}/search"
        params = {
            "q": artist_name,
            "type": "track",
            "limit": min(limit, 50)  # Spotify max is 50
        }
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 401:
            self._authenticate()
            response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("tracks", {}).get("items", [])
            
            # Filter tracks that match the artist name
            matching_tracks = []
            for track in items:
                artists = track.get("artists", [])
                artist_names = [a.get("name", "").lower() for a in artists]
                
                # Include if artist name is in the track's artists
                if artist_name.lower() in " ".join(artist_names) or any(artist_name.lower() in name for name in artist_names):
                    matching_tracks.append({
                        "track_id": track["id"],
                        "track_name": track["name"],
                        "album_name": track["album"]["name"],
                        "album_id": track["album"]["id"],
                        "popularity": track.get("popularity", 0) or 0,
                        "duration_ms": track.get("duration_ms", 0),
                        "preview_url": track.get("preview_url")
                    })
            
            return matching_tracks
        
        return []
    
    def collect_artist_data(self, artist_name: str) -> Dict:
        """Recolecta todos los datos de un artista"""
        stats = self.get_artist_stats(artist_name)
        
        if "error" in stats:
            return stats
        
        tracks = []
        
        # Collect tracks from albums
        for album in stats.get("recent_albums", []):
            album_tracks = self.get_album_tracks(album["album_id"])
            for album_track in album_tracks:
                tracks.append({
                    "track_id": album_track["track_id"],
                    "artist_name": stats["artist_name"],
                    "track_name": album_track["track_name"],
                    "album_name": album["album_name"],
                    "release_date": album["release_date"],
                    "track_position": album_track.get("track_number", 1)
                })
        
        # Sort by track position (earlier tracks in album are more important)
        tracks.sort(key=lambda x: x.get("track_position", 1))
        
        return {
            "artist_name": artist_name,
            "spotify_name": stats.get("artist_name"),
            "artist_id": stats.get("artist_id"),
            "followers": stats.get("followers", 0),
            "spotify_popularity": stats.get("popularity", 0),
            "genres": stats.get("genres", []),
            "tracks": tracks,
            "albums": stats.get("recent_albums", []),
            "_name_warning": stats.get("_name_warning")
        }


# Nota: Spotify no proporciona streams históricos vía API estándar
# Para datos históricos reales, se necesita Spotify for Artists API
# o datos manuales/CSV
# Esta implementación guarda lo que se puede obtener (tracks, popularity)
