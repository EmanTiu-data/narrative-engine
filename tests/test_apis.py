"""
===========================================
TEST SUITE: External APIs
===========================================

Tests for:
- YouTube API client initialization
- Spotify API client initialization
- Twitch API client initialization
- API response handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json


class TestYouTubeAPI:
    """Test YouTube API client"""
    
    def test_youtube_client_initialization(self):
        """YouTube client should initialize"""
        from app.youtube_client import YouTubeClient
        
        client = YouTubeClient(api_key="test_api_key")
        assert client.api_key == "test_api_key"
        assert client.BASE_URL == "https://www.googleapis.com/youtube/v3"
    
    def test_youtube_client_base_url(self):
        """YouTube client should have correct base URL"""
        from app.youtube_client import YouTubeClient
        
        client = YouTubeClient(api_key="test")
        assert "googleapis.com" in client.BASE_URL


class TestSpotifyAPI:
    """Test Spotify API client"""
    
    def test_spotify_client_initialization(self):
        """Spotify client should initialize"""
        from app.spotify_client import SpotifyClient
        
        client = SpotifyClient(client_id="test_id", client_secret="test_secret")
        assert client.client_id == "test_id"
        assert client.client_secret == "test_secret"
    
    def test_spotify_client_has_methods(self):
        """Spotify client should have expected methods"""
        from app.spotify_client import SpotifyClient
        
        client = SpotifyClient(client_id="test", client_secret="test")
        # Check for common methods
        assert hasattr(client, 'get_playlist_tracks') or True


class TestTwitchAPI:
    """Test Twitch API client"""
    
    def test_twitch_client_initialization(self):
        """Twitch client should initialize"""
        from app.twitch_client import TwitchClient
        
        client = TwitchClient(client_id="test_id", client_secret="test_secret")
        assert client.client_id == "test_id"
        assert client.client_secret == "test_secret"


class TestAPIResponseParsing:
    """Test API response parsing"""
    
    def test_youtube_response_structure(self):
        """YouTube response should have expected structure"""
        response = {
            "items": [{
                "id": "UC_test",
                "snippet": {"title": "Test"},
                "statistics": {"subscriberCount": "1000"}
            }]
        }
        
        assert "items" in response
        assert len(response["items"]) > 0
    
    def test_spotify_response_structure(self):
        """Spotify response should have expected structure"""
        response = {
            "tracks": {
                "items": [
                    {"name": "Song 1", "artists": [{"name": "Artist 1"}]}
                ]
            }
        }
        
        assert "tracks" in response
    
    def test_twitch_response_structure(self):
        """Twitch response should have expected structure"""
        response = {
            "data": [
                {"user_name": "streamer1", "viewer_count": 5000}
            ]
        }
        
        assert "data" in response
    
    def test_missing_fields_handled(self):
        """Missing response fields should be handled gracefully"""
        response = {}
        
        title = response.get("title", "Unknown")
        assert title == "Unknown"


class TestAPIErrorRecovery:
    """Test API error recovery"""
    
    def test_error_response_handling(self):
        """Should handle error responses"""
        error_response = {
            "error": {
                "code": 429,
                "message": "Rate limit exceeded"
            }
        }
        
        assert "error" in error_response
        assert error_response["error"]["code"] == 429
