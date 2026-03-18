"""
Tests for Insights Engine
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.insights_engine import (
    InsightsEngine,
    calculate_rating,
    get_rating_badge,
    get_quick_insight
)


class TestRatingSystem:
    """Tests for the rating system."""
    
    def test_calculate_rating_S(self):
        assert calculate_rating(600) == "S"
        assert calculate_rating(500) == "S"
    
    def test_calculate_rating_A_plus(self):
        assert calculate_rating(300) == "A+"
        assert calculate_rating(200) == "A+"
    
    def test_calculate_rating_A(self):
        assert calculate_rating(150) == "A"
        assert calculate_rating(100) == "A"
    
    def test_calculate_rating_B(self):
        assert calculate_rating(75) == "B"
        assert calculate_rating(50) == "B"
    
    def test_calculate_rating_C(self):
        assert calculate_rating(25) == "C"
        assert calculate_rating(0) == "C"
    
    def test_calculate_rating_D(self):
        assert calculate_rating(-10) == "D"
        assert calculate_rating(-50) == "D"
        assert calculate_rating(-100) == "D"
    
    def test_get_rating_badge(self):
        assert get_rating_badge("S") == "🟢"
        assert get_rating_badge("A+") == "🟢"
        assert get_rating_badge("A") == "🟡"
        assert get_rating_badge("B") == "🟠"
        assert get_rating_badge("C") == "🔴"
        assert get_rating_badge("D") == "🔴"


class TestEngagementCalculation:
    """Tests for engagement calculation."""
    
    def test_calculate_basic_engagement(self):
        engine = InsightsEngine()
        # (1000 + 100*2) / 10000 * 100 = 12%
        engagement = engine.calculate_engagement(10000, 1000, 100)
        assert engagement == 12.0
    
    def test_calculate_engagement_zero_views(self):
        engine = InsightsEngine()
        engagement = engine.calculate_engagement(0, 100, 50)
        assert engagement == 0.0
    
    def test_calculate_high_comments_engagement(self):
        engine = InsightsEngine()
        # Comments are weighted 2x
        engagement = engine.calculate_engagement(10000, 100, 1000)
        # (100 + 1000*2) / 10000 * 100 = 21%
        assert engagement == 21.0


class TestInsightsGeneration:
    """Tests for insight generation."""
    
    def test_explain_high_engagement(self):
        engine = InsightsEngine()
        # Calculate engagement: (50000 + 20000*2) / 500000 * 100 = 18%
        # Average engagement: 6%
        # Change: (18 - 6) / 6 * 100 = 200%
        result = engine.explain_engagement(
            views=500000,
            likes=50000,
            comments=20000,
            avg_views=200000,
            avg_likes=10000,
            avg_comments=2000,
            avg_engagement=6.0
        )
        
        assert result["rating"] in ["S", "A+", "A"]
        assert "insight_text" in result
        assert len(result["drivers"]) > 0
    
    def test_explain_low_engagement(self):
        engine = InsightsEngine()
        result = engine.explain_engagement(
            views=50000,
            likes=500,
            comments=100,
            avg_views=200000,
            avg_likes=10000,
            avg_comments=2000,
            avg_engagement=6.0
        )
        
        assert result["rating"] in ["C", "D"]
        assert "insight_text" in result
    
    def test_identify_drivers(self):
        engine = InsightsEngine()
        drivers = engine.identify_drivers(
            likes=50000,
            comments=10000,
            views=100000,
            avg_likes=10000,
            avg_comments=1000,
            avg_views=200000
        )
        
        assert len(drivers) > 0
        assert any("alto ratio" in d or "likes" in d or "comentarios" in d for d in drivers)


class TestAnomalyExplanation:
    """Tests for anomaly explanation."""
    
    def test_explain_spike(self):
        engine = InsightsEngine()
        result = engine.explain_anomaly(
            metric="vistas",
            value=1000000,
            mean=200000,
            std=100000,
            z_score=8.0
        )
        
        assert result["is_spike"] == True
        assert "insight_text" in result
        assert "⚠️" in result["insight_text"]
    
    def test_explain_drop(self):
        engine = InsightsEngine()
        result = engine.explain_anomaly(
            metric="likes",
            value=100,
            mean=1000,
            std=200,
            z_score=-4.5
        )
        
        assert result["is_spike"] == False
        assert "insight_text" in result


class TestVideoInsightGeneration:
    """Tests for full video insight generation."""
    
    def test_generate_video_insight_complete(self):
        engine = InsightsEngine()
        
        video_data = {
            "video_id": "test123",
            "title": "Test Video",
            "views": 500000,
            "likes": 25000,
            "comments_count": 5000,
            "published_at": "2024-01-15T15:00:00Z"
        }
        
        channel_avg = {
            "avg_views": 200000,
            "avg_likes": 10000,
            "avg_comments": 2000,
            "avg_engagement": 6.0
        }
        
        insight = engine.generate_video_insight(
            video_data=video_data,
            channel_avg=channel_avg,
            top_topic="tutorial"
        )
        
        assert insight["video_id"] == "test123"
        assert insight["title"] == "Test Video"
        assert "rating" in insight
        assert "insight_text" in insight
        assert "engagement_score" in insight
    
    def test_format_insight_for_display(self):
        engine = InsightsEngine()
        
        insight = {
            "rating": "A+",
            "rating_badge": "🟢",
            "insight_text": "250% más engagement que promedio. Drivers: alto ratio comments/vistas, likes elevados.",
            "drivers": ["alto ratio comments/vistas", "likes elevados"],
            "anomalies": [],
            "tip": "Tu horario de publicación es óptimo"
        }
        
        formatted = engine.format_insight_for_display(insight)
        
        assert "250% más engagement" in formatted
        assert "Drivers:" in formatted
        assert "Tip:" in formatted


class TestQuickInsight:
    """Tests for the convenience function."""
    
    def test_get_quick_insight(self):
        insight = get_quick_insight(
            views=500000,
            likes=25000,
            comments=5000
        )
        
        assert isinstance(insight, str)
        assert len(insight) > 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
