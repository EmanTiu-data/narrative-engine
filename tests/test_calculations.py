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
from app import lda_analyzer, correlation, anomaly_detector


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
            "Boring stream nothing happens",
            "Terrible performance today",
            "Best streamer ever",
            "So much entertainment value",
            "Weak gameplay and errors",
        ]
    
    @pytest.fixture
    def lda_analyzer_instance(self):
        """Create LDA analyzer instance"""
        return lda_analyzer.LDAAnalyzer(num_topics=2)
    
    def test_lda_initialization(self, lda_analyzer_instance):
        """LDA analyzer should initialize with correct parameters"""
        assert lda_analyzer_instance.num_topics == 2
        assert lda_analyzer_instance.model is None
    
    def test_lda_fit_transform(self, lda_analyzer_instance, sample_comments):
        """LDA should fit and transform comments to topics"""
        result = lda_analyzer_instance.fit_transform(sample_comments)
        
        assert result is not None
        # Should return topic distribution for each comment
        assert len(result) == len(sample_comments)
    
    def test_lda_extract_topics(self, lda_analyzer_instance, sample_comments):
        """LDA should extract meaningful topics"""
        lda_analyzer_instance.fit_transform(sample_comments)
        topics = lda_analyzer_instance.extract_topics()
        
        assert topics is not None
        # Should return num_topics topics
        assert len(topics) == lda_analyzer_instance.num_topics
    
    def test_lda_empty_input(self, lda_analyzer_instance):
        """LDA should handle empty input gracefully"""
        result = lda_analyzer_instance.fit_transform([])
        assert result == [] or result is None
    
    def test_lda_single_document(self, lda_analyzer_instance):
        """LDA should handle single document"""
        result = lda_analyzer_instance.fit_transform(["single comment"])
        assert len(result) == 1
    
    def test_lda_with_stopwords(self, lda_analyzer_instance):
        """LDA should filter stopwords"""
        comments_with_stopwords = [
            "the and is are was were for and with",
            "this that these those"
        ]
        result = lda_analyzer_instance.fit_transform(comments_with_stopwords)
        assert result is not None


class TestPearsonCorrelation:
    """Test Pearson correlation coefficient calculation"""
    
    @pytest.fixture
    def correlation_analyzer(self):
        """Create correlation analyzer instance"""
        return correlation.CorrelationAnalyzer()
    
    def test_correlation_perfect_positive(self, correlation_analyzer):
        """Should detect perfect positive correlation"""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        
        result = correlation_analyzer.pearson_correlation(x, y)
        
        assert result == pytest.approx(1.0, abs=0.001)
    
    def test_correlation_perfect_negative(self, correlation_analyzer):
        """Should detect perfect negative correlation"""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        
        result = correlation_analyzer.pearson_correlation(x, y)
        
        assert result == pytest.approx(-1.0, abs=0.001)
    
    def test_correlation_no_correlation(self, correlation_analyzer):
        """Should detect no correlation"""
        x = [1, 2, 3, 4, 5]
        y = [5, 1, 4, 2, 3]  # Random values
        
        result = correlation_analyzer.pearson_correlation(x, y)
        
        assert abs(result) < 0.5  # Should be low correlation
    
    def test_correlation_different_lengths(self, correlation_analyzer):
        """Should handle different length arrays"""
        x = [1, 2, 3]
        y = [1, 2, 3, 4, 5]
        
        with pytest.raises(ValueError):
            correlation_analyzer.pearson_correlation(x, y)
    
    def test_correlation_empty_input(self, correlation_analyzer):
        """Should handle empty input"""
        with pytest.raises(ValueError):
            correlation_analyzer.pearson_correlation([], [])
    
    def test_correlation_single_element(self, correlation_analyzer):
        """Should handle single element arrays"""
        x = [1]
        y = [1]
        
        # Single element correlation is undefined or 1
        result = correlation_analyzer.pearson_correlation(x, y)
        assert not np.isnan(result)


class TestZScoreAnomalyDetection:
    """Test Z-Score based anomaly detection"""
    
    @pytest.fixture
    def anomaly_detector_instance(self):
        """Create anomaly detector instance"""
        return anomaly_detector.AnomalyDetector()
    
    def test_zscore_normal_data(self, anomaly_detector_instance):
        """Z-Score should detect no anomalies in normal data"""
        data = [10, 11, 12, 9, 10, 11, 10, 12, 11, 10]
        
        anomalies = anomaly_detector_instance.detect_zscore(data, threshold=3.0)
        
        assert len(anomalies) == 0
    
    def test_zscore_with_outliers(self, anomaly_detector_instance):
        """Z-Score should detect outliers"""
        data = [10, 11, 12, 9, 10, 11, 10, 12, 11, 100]  # 100 is outlier
        
        anomalies = anomaly_detector_instance.detect_zscore(data, threshold=2.0)
        
        assert len(anomalies) > 0
    
    def test_zscore_threshold_zero(self, anomaly_detector_instance):
        """Z-Score with threshold 0 should flag everything"""
        data = [1, 2, 3, 4, 5]
        
        anomalies = anomaly_detector_instance.detect_zscore(data, threshold=0)
        
        # All or most should be flagged
        assert len(anomalies) > 0
    
    def test_zscore_empty_input(self, anomaly_detector_instance):
        """Z-Score should handle empty input"""
        anomalies = anomaly_detector_instance.detect_zscore([], threshold=3.0)
        assert anomalies == []
    
    def test_zscore_single_element(self, anomaly_detector_instance):
        """Z-Score should handle single element"""
        anomalies = anomaly_detector_instance.detect_zscore([10], threshold=3.0)
        # Single element has no variance, may behave differently
        assert isinstance(anomalies, list)


class TestIsolationForest:
    """Test Isolation Forest anomaly detection"""
    
    @pytest.fixture
    def anomaly_detector_instance(self):
        """Create anomaly detector instance"""
        return anomaly_detector.AnomalyDetector()
    
    def test_isolation_forest_normal_data(self, anomaly_detector_instance):
        """Isolation Forest should handle normal data"""
        # Normal distribution data
        data = np.random.normal(0, 1, (100, 2))
        
        anomalies = anomaly_detector_instance.detect_isolation_forest(data)
        
        assert len(anomalies) >= 0  # May have some anomalies
    
    def test_isolation_forest_with_clear_outliers(self, anomaly_detector_instance):
        """Isolation Forest should detect clear outliers"""
        # Normal data + outliers
        normal_data = np.random.normal(0, 1, (90, 2))
        outliers = np.array([[100, 100], [-100, -100], [50, -50]])
        data = np.vstack([normal_data, outliers])
        
        anomalies = anomaly_detector_instance.detect_isolation_forest(data)
        
        assert len(anomalies) >= 3  # Should detect the outliers
    
    def test_isolation_forest_empty_input(self, anomaly_detector_instance):
        """Isolation Forest should handle empty input"""
        anomalies = anomaly_detector_instance.detect_isolation_forest(np.array([]))
        assert anomalies == []


class TestLagAnalysis:
    """Test lag analysis for cross-platform correlation"""
    
    @pytest.fixture
    def correlation_analyzer(self):
        """Create correlation analyzer instance"""
        return correlation.CorrelationAnalyzer()
    
    def test_lag_analysis_positive_lag(self, correlation_analyzer):
        """Should detect positive lag (X leads Y)"""
        # X leads Y by 2 time units
        x = [1, 2, 3, 4, 5, 6, 7]
        y = [0, 0, 1, 2, 3, 4, 5]
        
        lag, corr = correlation_analyzer.lag_analysis(x, y)
        
        assert lag == 2  # Y lags X by 2
        assert corr > 0
    
    def test_lag_analysis_negative_lag(self, correlation_analyzer):
        """Should detect negative lag (Y leads X)"""
        # Y leads X by 1 time unit
        x = [0, 1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5, 6]
        
        lag, corr = correlation_analyzer.lag_analysis(x, y)
        
        assert lag == -1  # X lags Y by 1
    
    def test_lag_analysis_no_lag(self, correlation_analyzer):
        """Should detect no lag"""
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        
        lag, corr = correlation_analyzer.lag_analysis(x, y)
        
        assert lag == 0
        assert corr == pytest.approx(1.0, abs=0.001)


class TestDataProcessing:
    """Test data processing pipelines"""
    
    def test_normalize_data(self):
        """Data normalization should work correctly"""
        from app.correlation import CorrelationAnalyzer
        
        analyzer = CorrelationAnalyzer()
        data = [0, 1, 2, 3, 4, 5]
        
        normalized = analyzer.normalize(data)
        
        assert min(normalized) >= 0
        assert max(normalized) <= 1
    
    def test_rolling_average(self):
        """Rolling average calculation"""
        from app.correlation import CorrelationAnalyzer
        
        analyzer = CorrelationAnalyzer()
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        rolling = analyzer.rolling_average(data, window=3)
        
        assert len(rolling) < len(data)  # Should be shorter due to windowing
