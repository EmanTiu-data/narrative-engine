"""
Insights Engine for Narrative Intelligence Engine
Genera explicaciones narrativas estilo analista para videos y canales.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json


# ============================================================================
# TEMPLATES NARRATIVOS (Brief - 2-3 líneas)
# ============================================================================

ENGAGEMENT_TEMPLATES = {
    "very_high": "{pct}% más engagement que promedio del canal. Drivers: {drivers}. Top topic: {top_topic}.",
    "high": "{pct}% más engagement que promedio. Drivers: {drivers}.",
    "average": "Engagement promedio ({pct}% vs media). Top topic: {top_topic}.",
    "low": "{pct}% bajo promedio. Causa probable: {cause}. Tip: {tip}.",
    "very_low": "Engagement significativamente bajo ({pct}% vs promedio). Tip: {tip}."
}

ANOMALY_TEMPLATES = {
    "spike": "⚠️ ANOMALÍA: {metric} {sigma}σ sobre promedio ({value}). Causa: {cause}",
    "drop": "⚠️ CAÍDA: {metric} {sigma}σ bajo promedio ({value}). Causa: {cause}",
    "isolated": "🔍 OUTLIER: {metric} = {value} detectado por Isolation Forest."
}

CORRELATION_TEMPLATES = {
    "strong_positive": "📊 {platform_a} ↔ {platform_b}: correlación fuerte (r={r}). {interpretation}",
    "moderate_positive": "📊 {platform_a} ↔ {platform_b}: correlación moderada (r={r}). {interpretation}",
    "weak": "📊 {platform_a} ↔ {platform_b}: correlación débil (r={r}). Sin patrón claro.",
    "negative": "📊 {platform_a} ↔ {platform_b}: correlación negativa (r={r}). {interpretation}"
}

# ============================================================================
# RATING SYSTEM
# ============================================================================

RATING_THRESHOLDS = {
    "S": 500,      # >=500% above average
    "A+": 200,     # >=200% above average
    "A": 100,      # >=100% above average
    "B": 50,       # >=50% above average
    "C": 0,        # >=0% above average
    "D": -999999   # everything else (below average)
}

RATING_COLORS = {
    "S": "🟢",
    "A+": "🟢",
    "A": "🟡",
    "B": "🟠",
    "C": "🔴",
    "D": "🔴"
}


def calculate_rating(pct_change: float) -> str:
    """Calcula el rating basado en el porcentaje de cambio."""
    if pct_change >= RATING_THRESHOLDS["S"]:
        return "S"
    elif pct_change >= RATING_THRESHOLDS["A+"]:
        return "A+"
    elif pct_change >= RATING_THRESHOLDS["A"]:
        return "A"
    elif pct_change >= RATING_THRESHOLDS["B"]:
        return "B"
    elif pct_change >= RATING_THRESHOLDS["C"]:
        return "C"
    else:
        return "D"


def get_rating_badge(rating: str) -> str:
    """Retorna el badge visual para un rating."""
    return RATING_COLORS.get(rating, "⚪")


# ============================================================================
# INSIGHTS ENGINE
# ============================================================================

class InsightsEngine:
    """
    Motor de insights on-demand para videos y canales.
    
    Genera explicaciones narrativas estilo analista (2-3 líneas)
    explicando POR QUÉ un video tiene cierto engagement o anomalía.
    """
    
    def __init__(self, db=None):
        """
        Inicializa el motor de insights.
        
        Args:
            db: Instancia de DatabaseManager (opcional, para cachear insights)
        """
        self.db = db
    
    def calculate_engagement(self, views: int, likes: int, comments: int) -> float:
        """
        Calcula el engagement score.
        
        Args:
            views: Número de vistas
            likes: Número de likes
            comments: Número de comentarios
            
        Returns:
            Engagement score como porcentaje
        """
        if views == 0:
            return 0.0
        return ((likes + comments * 2) / views) * 100
    
    def identify_drivers(self, likes: int, comments: int, views: int, 
                        avg_likes: float, avg_comments: float, avg_views: float) -> List[str]:
        """
        Identifica los principales drivers de engagement.
        
        Returns:
            Lista de drivers (e.g., ["likes", "ratio comments/vistas"])
        """
        drivers = []
        
        # Check likes ratio
        if views > 0 and comments / views > 0.03:
            drivers.append("alto ratio comments/vistas")
        
        # Check likes vs average
        if avg_likes > 0 and likes > avg_likes * 1.5:
            drivers.append("likes elevados")
        
        # Check comments vs average
        if avg_comments > 0 and comments > avg_comments * 1.5:
            drivers.append("alto volumen de comentarios")
        
        # Check views efficiency
        if avg_views > 0 and views > avg_views * 2:
            drivers.append("alcance superior")
        
        if not drivers:
            drivers.append("engagement balanceado")
        
        return drivers
    
    def explain_engagement(self, views: int, likes: int, comments: int,
                           avg_views: float, avg_likes: float, 
                           avg_comments: float, avg_engagement: float,
                           top_topic: Optional[str] = None,
                           publish_hour: Optional[int] = None,
                           publish_day: Optional[str] = None) -> Dict[str, Any]:
        """
        Genera explicación de engagement estilo analista.
        
        Args:
            views, likes, comments: Métricas del video
            avg_*: Promedios del canal
            top_topic: Topic dominante del video
            publish_hour: Hora de publicación (0-23)
            publish_day: Día de publicación (e.g., "Martes")
            
        Returns:
            Diccionario con analysis, rating y insight_text
        """
        # Calculate engagement
        engagement = self.calculate_engagement(views, likes, comments)
        
        # Calculate percentage change
        if avg_engagement > 0:
            pct_change = ((engagement - avg_engagement) / avg_engagement) * 100
        else:
            pct_change = 0
        
        # Get rating
        rating = calculate_rating(pct_change)
        badge = get_rating_badge(rating)
        
        # Identify drivers
        drivers = self.identify_drivers(likes, comments, views, avg_likes, avg_comments, avg_views)
        drivers_str = ", ".join(drivers) if drivers else "sin factores destacados"
        
        # Determine template based on engagement level
        if pct_change > 100:
            template_key = "very_high" if pct_change > 200 else "high"
            template = ENGAGEMENT_TEMPLATES[template_key]
        elif pct_change > -10:
            template_key = "average"
            template = ENGAGEMENT_TEMPLATES[template_key]
        else:
            template_key = "low" if pct_change > -50 else "very_low"
            template = ENGAGEMENT_TEMPLATES[template_key]
        
        # Generate tip
        tip = self._generate_tip(publish_hour, publish_day, top_topic, pct_change)
        
        # Generate cause for low engagement
        cause = self._identify_low_engagement_cause(views, likes, comments, 
                                                     avg_views, avg_likes, avg_comments)
        
        # Build insight text (without rating badge - that's shown separately)
        insight_text = template.format(
            pct=abs(int(pct_change)),
            drivers=drivers_str,
            top_topic=top_topic or "general",
            cause=cause,
            tip=tip
        )
        
        return {
            "engagement_score": round(engagement, 2),
            "pct_vs_average": round(pct_change, 1),
            "rating": rating,
            "rating_badge": badge,
            "drivers": drivers,
            "insight_text": insight_text,
            "tip": tip
        }
    
    def _generate_tip(self, hour: Optional[int], day: Optional[str],
                       topic: Optional[str], pct_change: float) -> str:
        """Genera un tip actionable basado en el contexto."""
        tips = []
        
        # Time-based tips
        if hour is not None:
            if 14 <= hour <= 17:
                tips.append("Tu horario de publicación es óptimo")
            elif hour < 12:
                tips.append("Considera publicar entre 2-5 PM para mayor reach")
            elif hour > 20:
                tips.append("Horarios nocturnos pueden limitar alcance")
        
        if day:
            day_lower = day.lower()
            if day_lower in ["martes", "miercoles", "jueves"]:
                tips.append(f"{day} es buen día para tu audiencia")
            elif day_lower in ["viernes", "sabado"]:
                tips.append("Fines de semana tienen menor retención para tutoriales")
        
        # Topic-based tips
        if topic and pct_change > 100:
            tips.append(f"Continúa produciendo contenido tipo '{topic}'")
        
        if not tips:
            tips.append("Ajusta frecuencia de publicación para encontrar tu ritmo")
        
        return ". ".join(tips)
    
    def _identify_low_engagement_cause(self, views: int, likes: int, comments: int,
                                       avg_views: float, avg_likes: float, 
                                       avg_comments: float) -> str:
        """Identifica la causa probable de bajo engagement."""
        if avg_views > 0 and views < avg_views * 0.5:
            return "bajo alcance (vistas reducidas)"
        if avg_likes > 0 and likes < avg_likes * 0.5:
            return "contenido poco resonante (bajo likes)"
        if avg_comments > 0 and comments < avg_comments * 0.5:
            return "baja interacción (pocos comentarios)"
        return "factores externos o timing desfavorable"
    
    def explain_anomaly(self, metric: str, value: float, mean: float, 
                       std: float, z_score: float) -> Dict[str, Any]:
        """
        Genera explicación de anomalía.
        
        Args:
            metric: Nombre de la métrica (views, likes, comments)
            value: Valor outlier
            mean: Promedio
            std: Desviación estándar
            z_score: Z-score calculado
            
        Returns:
            Diccionario con explicación de anomalía
        """
        if std == 0:
            sigma = 0
        else:
            sigma = abs(round(z_score, 1))
        
        # Determine if spike or drop
        if value > mean:
            template_key = "spike"
            template = ANOMALY_TEMPLATES[template_key]
            cause = self._explain_spike_cause(metric, value, mean, sigma)
        else:
            template_key = "drop"
            template = ANOMALY_TEMPLATES[template_key]
            cause = self._explain_drop_cause(metric, value, mean, sigma)
        
        insight_text = template.format(
            metric=metric,
            value=f"{value:,.0f}" if value > 100 else f"{value:.1f}",
            sigma=sigma,
            cause=cause
        )
        
        return {
            "metric": metric,
            "value": value,
            "sigma": sigma,
            "is_spike": value > mean,
            "insight_text": insight_text,
            "cause": cause
        }
    
    def _explain_spike_cause(self, metric: str, value: float, mean: float, sigma: float) -> str:
        """Explica causa de spike en métrica."""
        ratio = value / mean if mean > 0 else 1
        
        if metric == "views":
            if ratio > 5:
                return "viralidad probable o promoción externa"
            elif ratio > 2:
                return "contenido compartido orgánicamente"
            else:
                return "efecto residual de streams o collabs"
        
        elif metric == "likes":
            if ratio > 3:
                return "contenido muy resonante emocionalmente"
            else:
                return "audiencia muy comprometida"
        
        elif metric == "comments":
            if ratio > 3:
                return "contenido controversial o muy discutible"
            else:
                return "llamado a la acción efectivo"
        
        return "actividad anómala de la audiencia"
    
    def _explain_drop_cause(self, metric: str, value: float, mean: float, sigma: float) -> str:
        """Explica causa de drop en métrica."""
        if metric == "views":
            return "posible cambio de algoritmo o contenido fuera de nicho"
        elif metric == "likes":
            return "contenido que no conectó emocionalmente"
        elif metric == "comments":
            return "falta de engagement o temas poco discussible"
        return "regresión a la media"
    
    def generate_video_insight(self, video_data: Dict[str, Any], 
                                channel_avg: Optional[Dict[str, float]] = None,
                                top_topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Genera insight completo para un video.
        
        Args:
            video_data: Diccionario con keys 'views', 'likes', 'comments', 
                       'published_at', 'title', etc.
            channel_avg: Promedios del canal (opcional)
            top_topic: Topic LDA dominante (opcional)
            
        Returns:
            Insight completo con rating, drivers, explicación y tips
        """
        views = video_data.get("views", 0)
        likes = video_data.get("likes", 0)
        comments = video_data.get("comments_count", video_data.get("comments", 0))
        
        # Use channel averages if provided, otherwise estimate
        if channel_avg:
            avg_views = channel_avg.get("avg_views", views * 2)
            avg_likes = channel_avg.get("avg_likes", likes * 2)
            avg_comments = channel_avg.get("avg_comments", comments * 2)
            avg_engagement = channel_avg.get("avg_engagement", 2.0)
        else:
            # Estimate averages from the video itself (less accurate)
            avg_views = views * 1.5
            avg_likes = likes * 1.5
            avg_comments = comments * 1.5
            avg_engagement = self.calculate_engagement(avg_views, avg_likes, avg_comments)
        
        # Parse publish time if available
        publish_hour = None
        publish_day = None
        if video_data.get("published_at"):
            try:
                pub_dt = video_data["published_at"]
                if isinstance(pub_dt, str):
                    pub_dt = datetime.fromisoformat(pub_dt.replace("Z", "+00:00"))
                publish_hour = pub_dt.hour
                publish_day = pub_dt.strftime("%A")
            except:
                pass
        
        # Generate engagement explanation
        engagement_result = self.explain_engagement(
            views=views,
            likes=likes,
            comments=comments,
            avg_views=avg_views,
            avg_likes=avg_likes,
            avg_comments=avg_comments,
            avg_engagement=avg_engagement,
            top_topic=top_topic,
            publish_hour=publish_hour,
            publish_day=publish_day
        )
        
        # Check for anomalies
        anomalies = []
        if avg_views > 0 and abs(views - avg_views) > avg_views * 0.5:
            z_score = (views - avg_views) / avg_views if avg_views > 0 else 0
            anomaly = self.explain_anomaly("vistas", views, avg_views, avg_views, z_score)
            anomalies.append(anomaly)
        
        if avg_likes > 0 and abs(likes - avg_likes) > avg_likes * 0.5:
            z_score = (likes - avg_likes) / avg_likes if avg_likes > 0 else 0
            anomaly = self.explain_anomaly("likes", likes, avg_likes, avg_likes, z_score)
            anomalies.append(anomaly)
        
        return {
            "video_id": video_data.get("video_id", video_data.get("id", "unknown")),
            "title": video_data.get("title", "Unknown Title"),
            "views": views,
            "likes": likes,
            "comments": comments,
            **engagement_result,
            "anomalies": anomalies,
            "generated_at": datetime.now().isoformat()
        }
    
    def format_insight_for_display(self, insight: Dict[str, Any]) -> str:
        """
        Formatea el insight para display en dashboard.
        
        Args:
            insight: Resultado de generate_video_insight
            
        Returns:
            String formateado para mostrar
        """
        lines = []
        
        # Main insight text (already formatted)
        insight_text = insight.get('insight_text', '')
        if insight_text:
            lines.append(insight_text)
        
        # Drivers (if not already in insight text)
        drivers = insight.get("drivers", [])
        if drivers and "Drivers:" not in insight_text:
            lines.append(f"Drivers: {', '.join(drivers)}.")
        
        # Anomalies
        anomalies = insight.get("anomalies", [])
        for anomaly in anomalies:
            anomaly_text = anomaly.get("insight_text", "")
            if anomaly_text and anomaly_text not in insight_text:
                lines.append(anomaly_text)
        
        # Tip (if not already in insight text)
        tip = insight.get("tip", "")
        if tip and tip not in insight_text:
            lines.append(f"Tip: {tip}")
        
        return " ".join(lines)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_quick_insight(views: int, likes: int, comments: int) -> str:
    """
    Genera un insight rápido sin necesidad de instanciar la clase.
    
    Útil para uso rápido desde el dashboard.
    """
    engine = InsightsEngine()
    
    # Estimate average as 1.5x the values (rough approximation)
    avg_views = views * 1.5
    avg_likes = likes * 1.5
    avg_comments = comments * 1.5
    avg_engagement = engine.calculate_engagement(avg_views, avg_likes, avg_comments)
    
    result = engine.explain_engagement(
        views=views,
        likes=likes,
        comments=comments,
        avg_views=avg_views,
        avg_likes=avg_likes,
        avg_comments=avg_comments,
        avg_engagement=avg_engagement
    )
    
    return result["insight_text"]


# Example usage
if __name__ == "__main__":
    # Test with sample data
    engine = InsightsEngine()
    
    video = {
        "video_id": "test123",
        "title": "Tutorial Aim Perfecto",
        "views": 500000,
        "likes": 25000,
        "comments_count": 3200,
        "published_at": "2024-01-15T15:00:00Z"
    }
    
    channel_avg = {
        "avg_views": 200000,
        "avg_likes": 8000,
        "avg_comments": 800,
        "avg_engagement": 2.5
    }
    
    insight = engine.generate_video_insight(
        video_data=video,
        channel_avg=channel_avg,
        top_topic="tutorial/habilidad"
    )
    
    print("=" * 60)
    print(f"VIDEO: {insight['title']}")
    print(f"RATING: {insight['rating_badge']}{insight['rating']}")
    print(f"ENGAGEMENT: {insight['engagement_score']}%")
    print("=" * 60)
    print(f"\n💡 INSIGHT:")
    print(engine.format_insight_for_display(insight))
    print()
