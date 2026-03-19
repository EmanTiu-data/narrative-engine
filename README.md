# The Narrative Intelligence Engine

🧠 Advanced NLP, Cross-Platform Correlation & Anomaly Detection

## 🎯 Objective

Este proyecto nació de la necesidad de entender **por qué** ciertos creadores de contenido prosperan mientras otros no, analisando patrones en múltiples plataformas (YouTube, Spotify, Twitch).

Como analista de datos, mi visión es construir un motor que:
1. **Extraiga conocimiento** de grandes volúmenes de comentarios y métricas
2. **Identifique correlaciones** entre plataformas para predecir tendencias
3. **Detecte anomalías** que señalen oportunidades o problemas potenciales

---

## 🚀 Quick Start

```bash
# 1. Clonar el repositorio
git clone https://github.com/EmanTiu-data/narrative-engine.git
cd narrative-engine

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar datos NLTK (automático, pero puedes hacerlo manual)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 5. Configurar API keys
cp .env.example .env  # Si existe .env.example
# Editar .env con tus credenciales

# 6. Correr el dashboard
streamlit run app/dashboard.py
```

---

## 📋 Requisitos

- **Python 3.11+**
- APIs de YouTube, Spotify y Twitch (crea tus credenciales en los开发者 portals)

### APIs necesarias

| Plataforma | Portal de Desarrollo | Permisos |
|------------|---------------------|----------|
| YouTube | [Google Cloud Console](https://console.cloud.google.com/) | YouTube Data API v3 |
| Spotify | [Spotify Developer](https://developer.spotify.com/dashboard) | - |
| Twitch | [Twitch Dev Console](https://dev.twitch.tv/console) | - |

---

## 📁 Estructura del Proyecto

```
narrative-engine/
├── app/
│   ├── __init__.py          # Configuración del proyecto
│   ├── dashboard.py         # Interfaz Streamlit
│   ├── lda_analyzer.py      # Análisis de temas (NLP)
│   ├── correlation.py       # Correlación multi-plataforma
│   ├── anomaly_detector.py  # Detección de anomalías
│   ├── db.py                # Gestor de base de datos
│   ├── youtube_client.py    # Cliente YouTube API
│   ├── spotify_client.py    # Cliente Spotify API
│   └── twitch_client.py     # Cliente Twitch API
├── data/                    # Datos persists (auto-creado)
│   └── narrative.db        # SQLite database
├── tests/                   # Suite de tests
│   ├── conftest.py
│   ├── test_*.py
│   ├── edge_cases/
│   └── performance/
├── .github/
│   └── workflows/          # CI/CD pipelines
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

---

## 🔬 Features

### Fase 1: NLP Avanzado (LDA)

Extrae automáticamente los temas principales de miles de comentarios:

- **Topic Extraction**: Identifica hasta N temas usando Latent Dirichlet Allocation
- **Automatic Labeling**: Clasifica topics en categorías (skill, entertainment, frequency, etc.)
- **Ejemplo**: `"40% hablan sobre habilidad técnica"` → extraído automáticamente

```python
from app.lda_analyzer import LDAAnalyzer

analyzer = LDAAnalyzer(n_topics=5)
topics = analyzer.fit_transform(comments_list)
print(topics)
```

### Fase 2: Correlación Multi-Plataforma

Descubre relaciones entre métricas de diferentes plataformas:

- **Pearson Correlation**: Mide fuerza y dirección de relaciones lineales
- **Lag Analysis**: ¿Un pico en Twitch predice un pico en YouTube +3 días?
- **Graceful Degradation**: Funciona incluso si una plataforma falla

```python
from app.correlation import CorrelationEngine

engine = CorrelationEngine()
result = engine.calculate_pearson(youtube_series, twitch_series)
print(f"Correlation: {result['r']:.3f}")
```

### Fase 3: Detección de Anomalías

Identifica outliers y patrones inusuales:

- **Z-Score**: Detecta valores a más de 3σ de la media
- **Isolation Forest**: Algoritmo de ML para patrones complejos
- **Combined Detection**: Combina múltiples métodos para mayor precisión

```python
from app.anomaly_detector import AnomalyDetector

detector = AnomalyDetector()
result = detector.detect_combined(metrics_data)
print(f"Anomalías detectadas: {result['n_outliers']}")
```

---

## 💾 Persistencia de Datos

Los datos se almacenan automáticamente en SQLite:

```python
from app.db import DatabaseManager

db = DatabaseManager()
db.insert_youtube_video(video_data)
db.insert_topic(channel_name, topic_id, topic_name, distribution)
db.get_correlations()
```

La base de datos se crea automáticamente en `data/narrative.db`.

---

## 🧪 Testing

```bash
# Correr todos los tests
pytest tests/ -v

# Tests con coverage
pytest tests/ --cov=app --cov-report=html

# Solo tests de rendimiento
pytest tests/performance/ -v
```

---

## 🔧 Configuration

### Variables de Entorno (.env)

```env
# YouTube
YOUTUBE_API_KEY=tu_api_key_de_youtube

# Spotify
SPOTIFY_CLIENT_ID=tu_client_id
SPOTIFY_CLIENT_SECRET=tu_client_secret

# Twitch
TWITCH_CLIENT_ID=tu_client_id
TWITCH_CLIENT_SECRET=tu_client_secret
```

---

## 🛠️ Tech Stack

| Categoría | Tecnología | Propósito |
|-----------|------------|-----------|
| Dashboard | Streamlit | Interfaz web interactiva |
| NLP | scikit-learn, NLTK | Topic extraction y preprocessing |
| ML | scikit-learn | Isolation Forest, clustering |
| Stats | scipy.stats | Correlación de Pearson |
| Data | pandas, numpy | Manipulación de datos |
| Database | SQLite | Persistencia local |
| Visualization | Plotly | Gráficos interactivos |

---

## 📊 Dashboard

El dashboard incluye 4 tabs principales:

1. **Data Collection**: Recolecta datos de APIs
2. **Topic Analysis**: Visualiza topics extraídos con LDA
3. **Correlations**: Mapa de calor de correlaciones
4. **Anomalies**: Timeline con outliers marcados

---

## Contribuir

1. Fork el repositorio
2. Crea una branch (`git checkout -b feature/nueva-feature`)
3. Commit tus cambios (`git commit -m 'Add nueva feature'`)
4. Push a la branch (`git push origin feature/nueva-feature`)
5. Abre un Pull Request

---

## 📝 Licencia

MIT License - ver archivo LICENSE para detalles.

---

## 👤 Autor

**EmanTiu** - [@EmanTiu-data](https://github.com/EmanTiu-data)
