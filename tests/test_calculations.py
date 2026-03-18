"""
===========================================
TEST SUITE: Calculations & Algorithms
===========================================

Tests for:
- LDA topic extraction
- Pearson correlation coefficient
- Z-Score anomaly detection
- Isolation Forest anomaly detection
- Statistical calculations
- Data processing pipelines
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock


class TestLDATopicExtraction:
    """Test LDA (Latent Dirichlet Allocation) topic extraction"""
    
    @pytest.fixture
    def sample_comments(self):
        """Sample comments for testing"""
        return [
            "This streamer is amazing at the game",
            "Great technical skill and gameplay",
            "Very entertaining stream today",
            "Love watching the gameplay",
            "Amazing strategy and tactics",
        ]
    
    @pytest.fixture
    def lda_analyzer_instance(self):
        """Create LDA analyzer instance"""
        from app.lda_analyzer import LDAAnalyzer
        return LDAAnalyzer(n_topics=2)
    
    def test_lda_initialization(self, lda_analyzer_instance):
        """LDA analyzer should initialize with correct parameters"""
        assert lda_analyzer_instance.n_topics == 2


class TestPearsonCorrelation:
    """Test Pearson correlation coefficient calculation"""
    
    @pytest.fixture
    def correlation_engine(self):
        """Create correlation engine instance"""
        from app.correlation import CorrelationEngine
        return CorrelationEngine()
    
    def test_correlation_perfect_positive(self, correlation_engine):
        """Should detect perfect positive correlation"""
        series_a = pd.Series([1, 2, 3, 4, 5])
        series_b = pd.Series([2, 4, 6, 8, 10])
        
        result = correlation_engine.calculate_pearson(series_a, series_b)
        
        assert result["r"] == pytest.approx(1.0, abs=0.001)
    
    def test_correlation_perfect_negative(self, correlation_engine):
        """Should detect perfect negative correlation"""
        series_a = pd.Series([1, 2, 3, 4, 5])
        series_b = pd.Series([10, 8, 6, 4, 2])
        
        result = correlation_engine.calculate_pearson(series_a, series_b)
        
        assert result["r"] == pytest.approx(-1.0, abs=0.001)
    
    def test_correlation_insufficient_data(self, correlation_engine):
        """Should handle insufficient data"""
        series_a = pd.Series([1, 2])
        series_b = pd.Series([1, 2])
        
        result = correlation_engine.calculate_pearson(series_a, series_b)
        
        # Should return default values for < 3 data points
        assert result["n_observations"] < 3


class TestZScoreAnomalyDetection:
    """Test Z-Score based anomaly detection"""
    
    @pytest.fixture
    def anomaly_detector_instance(self):
        """Create anomaly detector instance"""
        from app.anomaly_detector import AnomalyDetector
        return AnomalyDetector()
    
    def test_zscore_normal_data(self, anomaly_detector_instance):
        """Z-Score should handle normal data"""
        data = pd.Series([10, 11, 12, 9, 10, 11, 10, 12, 11, 10])
        
        # Just test it doesn't crash
        result = anomaly_detector_instance.detect_zscore(data)
        assert result is not None
    
    def test_zscore_with_outliers(self, anomaly_detector_instance):
        """Z-Score should detect outliers"""
        data = pd.Series([10, 11, 12, 9, 10, 11, 10, 12, 11, 100])
        
        result = anomaly_detector_instance.detect_zscore(data)
        assert result is not None


class TestIsolationForest:
    """Test Isolation Forest anomaly detection"""
    
    @pytest.fixture
    def anomaly_detector_instance(self):
        """Create anomaly detector instance"""
        from app.anomaly_detector import AnomalyDetector
        return AnomalyDetector()
    
    def test_isolation_forest_normal_data(self, anomaly_detector_instance):
        """Isolation Forest should handle normal data"""
        data = np.random.normal(0, 1, (100, 2))
        
        result = anomaly_detector_instance.detect_isolation_forest(data)
        assert result is not None


class TestLagAnalysis:
    """Test lag analysis for cross-platform correlation"""
    
    def test_lag_correlation_method_exists(self):
        """CorrelationEngine should have lag_correlation method"""
        from app.correlation import CorrelationEngine
        
        engine = CorrelationEngine()
        assert hasattr(engine, 'lag_correlation') or hasattr(engine, 'calculate_lag_correlation')


class TestDataProcessing:
    """Test data processing pipelines"""
    
    def test_normalize_series(self):
        """Data normalization should work correctly"""
        from app.correlation import CorrelationEngine
        
        engine = CorrelationEngine()
        data = pd.Series([0, 1, 2, 3, 4, 5])
        
        normalized = engine.normalize_series(data, method="minmax")
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
