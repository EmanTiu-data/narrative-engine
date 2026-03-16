"""
Anomaly Detection Engine
Detecta outliers usando Z-Score e Isolation Forest
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """Detector de anomalías estadísticas"""
    
    def __init__(self, z_threshold: float = 3.0, contamination: float = 0.1):
        self.z_threshold = z_threshold
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.iso_forest = None
        self.fitted = False
    
    def detect_zscore(self, data: np.ndarray, threshold: float = None) -> Dict:
        """
        Detecta anomalías usando Z-Score
        Z-Score = (x - mean) / std
        Valores con |z| > threshold son anomalías
        """
        
        threshold = threshold or self.z_threshold
        
        if len(data) < 3:
            return {
                "error": "Se necesitan al menos 3 datos",
                "is_outlier": [],
                "z_scores": []
            }
        
        # Calculate z-scores
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return {
                "error": "Desviación estándar es 0",
                "is_outlier": [False] * len(data),
                "z_scores": [0] * len(data)
            }
        
        z_scores = (data - mean) / std
        is_outlier = np.abs(z_scores) > threshold
        
        return {
            "z_scores": z_scores.tolist(),
            "is_outlier": is_outlier.tolist(),
            "mean": mean,
            "std": std,
            "threshold": threshold,
            "n_outliers": int(np.sum(is_outlier)),
            "outlier_percentage": float(np.sum(is_outlier)) / len(data) * 100
        }
    
    def detect_iqr(self, data: np.ndarray, multiplier: float = 1.5) -> Dict:
        """
        Detecta anomalías usando IQR (Interquartile Range)
        Más robusto que Z-Score para datos no normales
        """
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        is_outlier = (data < lower_bound) | (data > upper_bound)
        
        return {
            "is_outlier": is_outlier.tolist(),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "n_outliers": int(np.sum(is_outlier)),
            "outlier_percentage": float(np.sum(is_outlier)) / len(data) * 100
        }
    
    def fit_isolation_forest(self, data: np.ndarray):
        """Entrena Isolation Forest"""
        
        # Reshape if needed
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Fit scaler
        scaled_data = self.scaler.fit_transform(data)
        
        # Fit isolation forest
        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.iso_forest.fit(scaled_data)
        self.fitted = True
    
    def detect_isolation_forest(self, data: np.ndarray) -> Dict:
        """
        Detecta anomalías usando Isolation Forest
        Mejor para patrones complejos y datos multidimensionales
        """
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        if not self.fitted:
            self.fit_isolation_forest(data)
        
        # Predict (-1 for outliers, 1 for inliers)
        scaled_data = self.scaler.transform(data)
        predictions = self.iso_forest.predict(scaled_data)
        scores = self.iso_forest.decision_function(scaled_data)
        
        is_outlier = predictions == -1
        
        return {
            "is_outlier": is_outlier.tolist(),
            "anomaly_scores": scores.tolist(),
            "n_outliers": int(np.sum(is_outlier)),
            "outlier_percentage": float(np.sum(is_outlier)) / len(data) * 100
        }
    
    def detect_combined(self, data: np.ndarray) -> Dict:
        """
        Detección combinada: Z-Score + Isolation Forest
        Solo marca como anomalía si ambos métodos concuerdan
        """
        
        zscore_result = self.detect_zscore(data)
        iso_result = self.detect_isolation_forest(data)
        
        # Combined: outlier if both agree or if Z-score is very extreme
        zscore_outliers = np.array(zscore_result.get("is_outlier", [False] * len(data)))
        iso_outliers = np.array(iso_result.get("is_outlier", [False] * len(data)))
        
        # Combined: require both methods to agree OR extreme Z-score (> 4 std)
        extreme_zscore = np.abs(np.array(zscore_result.get("z_scores", [0] * len(data)))) > 4
        
        is_outlier = (zscore_outliers & iso_outliers) | extreme_zscore
        
        return {
            "zscore": {
                "is_outlier": zscore_outliers.tolist(),
                "z_scores": zscore_result.get("z_scores", [])
            },
            "isolation_forest": iso_result,
            "combined": {
                "is_outlier": is_outlier.tolist(),
                "n_outliers": int(np.sum(is_outlier)),
                "outlier_percentage": float(np.sum(is_outlier)) / len(data) * 100
            }
        }
    
    def analyze_metric(self, data: np.ndarray, metric_name: str = "metric") -> Dict:
        """Analiza una métrica y detecta anomalías"""
        
        result = self.detect_combined(data)
        
        # Add summary
        if "error" not in result:
            outlier_indices = [i for i, x in enumerate(result["combined"]["is_outlier"]) if x]
            
            return {
                "metric": metric_name,
                "n_total": len(data),
                "n_outliers": result["combined"]["n_outliers"],
                "outlier_percentage": result["combined"]["outlier_percentage"],
                "outlier_indices": outlier_indices,
                "outlier_values": [float(data[i]) for i in outlier_indices] if outlier_indices else [],
                "zscore_method": {
                    "n_outliers": result["zscore"].get("n_outliers", 0),
                    "threshold": self.z_threshold
                },
                "isolation_forest": {
                    "n_outliers": result["isolation_forest"].get("n_outliers", 0),
                    "contamination": self.contamination
                },
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data))
            }
        
        return result


class VideoAnomalyDetector:
    """Detector específico para videos de YouTube"""
    
    def __init__(self, z_threshold: float = 2.0):
        self.detector = AnomalyDetector(z_threshold=z_threshold)
    
    def calculate_engagement(self, videos: List[Dict]) -> List[Dict]:
        """
        Calcula engagement score para cada video.
        Engagement = (likes + comments * 2) / views * 100
        """
        
        if not videos:
            return []
        
        # Calculate engagement for each video
        for video in videos:
            views = video.get("views", 0)
            likes = video.get("likes", 0)
            comments = video.get("comments_count", 0)
            
            if views > 0:
                # Engagement rate: likes + weighted comments / views
                engagement = ((likes + comments * 2) / views) * 100
            else:
                engagement = 0
            
            video["engagement_score"] = round(engagement, 2)
        
        return videos
    
    def rank_by_engagement(self, videos: List[Dict], top_n: int = 5) -> Dict:
        """
        Ordena videos por engagement y detecta "Interesantes" (20% acima del promedio)
        """
        
        if not videos:
            return {"error": "No videos provided"}
        
        # Calculate engagement scores
        videos = self.calculate_engagement(videos)
        
        # Calculate average engagement
        engagements = [v.get("engagement_score", 0) for v in videos]
        avg_engagement = np.mean(engagements)
        
        # Threshold for "interesting": 20% above average
        interesting_threshold = avg_engagement * 1.20
        
        # Mark videos as interesting or normal
        ranked_videos = []
        for video in videos:
            eng = video.get("engagement_score", 0)
            video["is_interesting"] = eng >= interesting_threshold
            video["engagement_vs_avg_pct"] = round(((eng - avg_engagement) / avg_engagement * 100) if avg_engagement > 0 else 0, 1)
            ranked_videos.append(video)
        
        # Sort by engagement (highest first)
        ranked_videos.sort(key=lambda x: x.get("engagement_score", 0), reverse=True)
        
        # Get top 5
        top_interesting = [v for v in ranked_videos if v.get("is_interesting")][:top_n]
        
        return {
            "total_videos": len(videos),
            "average_engagement": round(avg_engagement, 2),
            "interesting_threshold": round(interesting_threshold, 2),
            "interesting_count": len(top_interesting),
            "top_engaging": ranked_videos[:top_n],
            "top_interesting": top_interesting,
            "all_videos": ranked_videos
        }
    
    def analyze_video_metrics(self, videos: List[Dict]) -> Dict:
        """
        Analiza videos y detecta anomalías en métricas
        """
        
        if not videos:
            return {"error": "No videos provided"}
        
        # Extract metrics
        views = np.array([v.get("views", 0) for v in videos])
        likes = np.array([v.get("likes", 0) for v in videos])
        comments = np.array([v.get("comments_count", 0) for v in videos])
        
        # Analyze each metric
        views_result = self.detector.analyze_metric(views, "views")
        likes_result = self.detector.analyze_metric(likes, "likes")
        comments_result = self.detector.analyze_metric(comments, "comments")
        
        # Find outliers (any metric)
        all_outliers = set()
        
        for result in [views_result, likes_result, comments_result]:
            if "outlier_indices" in result:
                all_outliers.update(result["outlier_indices"])
        
        # Get outlier videos
        outlier_videos = []
        for idx in all_outliers:
            video = videos[idx].copy()
            video["index"] = idx
            
            # Add which metrics are anomalous
            video["anomalies"] = []
            if idx in views_result.get("outlier_indices", []):
                video["anomalies"].append(f"views ({views[idx]:,.0f})")
            if idx in likes_result.get("outlier_indices", []):
                video["anomalies"].append(f"likes ({likes[idx]:,.0f})")
            if idx in comments_result.get("outlier_indices", []):
                video["anomalies"].append(f"comments ({comments[idx]:,.0f})")
            
            outlier_videos.append(video)
        
        return {
            "total_videos": len(videos),
            "total_outliers": len(all_outliers),
            "outlier_percentage": len(all_outliers) / len(videos) * 100,
            "views": views_result,
            "likes": likes_result,
            "comments": comments_result,
            "outlier_videos": outlier_videos
        }
    
    def generate_alert(self, result: Dict) -> str:
        """Genera alerta para anomalías detectadas"""
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        lines = [
            "=" * 60,
            "⚠️ DETECCIÓN DE ANOMALÍAS",
            "=" * 60,
            f"Videos analizados: {result['total_videos']}",
            f"Anomalías detectadas: {result['total_outliers']} ({result['outlier_percentage']:.1f}%)",
            ""
        ]
        
        if result["outlier_videos"]:
            lines.append("📌 VIDEOS ANÓMALOS:")
            lines.append("-" * 40)
            
            for video in result["outlier_videos"]:
                lines.append(f"\n• {video.get('title', 'Sin título')[:50]}...")
                lines.append(f"  Anomalías: {', '.join(video.get('anomalies', []))}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def exclude_outliers(self, videos: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Excluye anomalías del dataset
        Returns: (normal_videos, outlier_videos)
        """
        
        result = self.analyze_video_metrics(videos)
        
        if "error" in result:
            return videos, []
        
        outlier_indices = set()
        for r in [result.get("views", {}), result.get("likes", {}), result.get("comments", {})]:
            outlier_indices.update(r.get("outlier_indices", []))
        
        normal_videos = [v for i, v in enumerate(videos) if i not in outlier_indices]
        outlier_videos = [v for i, v in enumerate(videos) if i in outlier_indices]
        
        return normal_videos, outlier_videos
