"""
Template Loader for Narrative Intelligence Engine
Carga las plantillas de historias desde archivos JSON.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class TemplateLoader:
    """
    Cargador de plantillas para generar narrativas.
    
    Separa la lógica de generación de historias de las plantillas,
    facilitando la personalización sin modificar código.
    """
    
    _instance: Optional['TemplateLoader'] = None
    _templates: Optional[Dict[str, Any]] = None
    
    def __new__(cls) -> 'TemplateLoader':
        """Singleton pattern para evitar cargar archivos múltiples veces."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Inicializa el cargador de plantillas."""
        if self._templates is None:
            self._load_templates()
    
    def _load_templates(self) -> None:
        """Carga las plantillas desde el archivo JSON."""
        template_path = Path(__file__).parent / "templates" / "story_templates.json"
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                self._templates = json.load(f)
        except FileNotFoundError:
            # Fallback a plantillas vacías si no existe el archivo
            self._templates = {
                "topic_labels": {},
                "correlation_templates": {},
                "anomaly_templates": {}
            }
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in template file: {e}")
    
    @property
    def topic_labels(self) -> Dict[str, Any]:
        """Obtiene las etiquetas de temas."""
        return self._templates.get("topic_labels", {})
    
    @property
    def correlation_templates(self) -> Dict[str, Any]:
        """Obtiene las plantillas de correlación."""
        return self._templates.get("correlation_templates", {})
    
    @property
    def anomaly_templates(self) -> Dict[str, Any]:
        """Obtiene las plantillas de anomalías."""
        return self._templates.get("anomaly_templates", {})
    
    def get_correlation_story(self, r: float, platform_a: str, platform_b: str,
                               metric_a: str, metric_b: str, 
                               lag_days: Optional[int] = None) -> str:
        """
        Genera una historia de correlación basada en los datos.
        
        Args:
            r: Coeficiente de correlación
            platform_a: Nombre de la primera plataforma
            platform_b: Nombre de la segunda plataforma
            metric_a: Métrica de la primera plataforma
            metric_b: Métrica de la segunda plataforma
            lag_days: Días de lag (si aplica)
            
        Returns:
            String con la historia formateada
        """
        if lag_days is not None and abs(r) >= 0.5:
            template = self.correlation_templates.get("lag_detected", {}).get("template", "")
            return template.format(
                days=lag_days,
                platform_a=platform_a,
                platform_b=platform_b
            )
        
        if r >= 0.7:
            template = self.correlation_templates.get("positive_strong", {}).get("template", "")
        elif r >= 0.4:
            template = self.correlation_templates.get("positive_moderate", {}).get("template", "")
        elif r <= -0.7:
            template = self.correlation_templates.get("negative_strong", {}).get("template", "")
        else:
            template = "Las plataformas {platform_a} y {platform_b} no muestran correlación significativa."
            return template.format(platform_a=platform_a, platform_b=platform_b)
        
        return template.format(
            platform_a=platform_a,
            platform_b=platform_b,
            value=f"{r:.2f}",
            metric_a=metric_a,
            metric_b=metric_b
        )
    
    def reload(self) -> None:
        """Recarga las plantillas desde el archivo."""
        self._load_templates()


# Instancia global para uso directo
templates = TemplateLoader()
