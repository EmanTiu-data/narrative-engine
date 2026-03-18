"""
LDA Topic Analyzer
Extrae temas latentes de comentarios usando Latent Dirichlet Allocation (LDA).

Este módulo proporciona capacidades de NLP para identificar automáticamente
temas y patrones en grandes volúmenes de texto, ideal para analizar
comentarios de YouTube, Twitch y otras plataformas.

Ejemplo de uso:
    analyzer = LDAAnalyzer(n_topics=5)
    topics = analyzer.fit_transform(comments_list)
"""

import re
import nltk
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class LDAAnalyzer:
    """
    Analizador de temas usando Latent Dirichlet Allocation (LDA).
    
    Identifica automáticamente los temas principales en una colección
    de documentos de texto mediante aprendizaje no supervisado.
    
    Attributes:
        n_topics: Número de topics a identificar
        n_top_words: Número de palabras por topic
        fitted: Indica si el modelo ya fue entrenado
    """
    
    def __init__(self, n_topics: int = 5, n_top_words: int = 10):
        """
        Inicializa el analizador LDA.
        
        Args:
            n_topics: Número de temas a identificar (default: 5)
            n_top_words: Número de palabras representativas por tema (default: 10)
        """
        # Validate parameters
        if n_topics < 1:
            raise ValueError("n_topics must be at least 1")
        if n_top_words < 1:
            raise ValueError("n_top_words must be at least 1")
            
        self.n_topics = n_topics
        self.n_top_words = n_top_words
        self.lemmatizer = WordNetLemmatizer()
        
        # Spanish + English stopwords (common in gaming/streaming context)
        self.stop_words = set(stopwords.words('spanish')) | set(stopwords.words('english'))
        
        # Add custom stopwords (common in comments)
        custom_stops = {
            'video', 'videos', 'stream', 'streams', 'youtube', 'twitch',
            'si', 'no', 'pero', 'mas', 'menos', 'tan', 'muy', 'mucho',
            'https', 'http', 'www', 'com', 'el', 'la', 'los', 'las',
            'un', 'una', 'unos', 'unas', 'es', 'son', 'está', 'están',
            'que', 'qué', 'de', 'del', 'en', 'con', 'por', 'para',
            'this', 'that', 'the', 'and', 'is', 'are', 'was', 'were',
            'like', 'just', 'get', 'got', 'go', 'going', 'one', 'would',
            'could', 'should', 'really', 'also', 'even', 'much', 'more'
        }
        self.stop_words.update(custom_stops)
        
        # Vectorizer
        self.vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=1000,
            ngram_range=(1, 2),
            stop_words=list(self.stop_words)
        )
        
        # LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=20,
            learning_method='online',
            random_state=42,
            n_jobs=-1
        )
        
        self.fitted = False
        self.feature_names = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocesa el texto para análisis LDA.
        
        Limpia el texto eliminando URLs, caracteres especiales y aplicando
        lematización para normalizar las palabras.
        
        Args:
            text: Texto raw a procesar
            
        Returns:
            Texto preprocesado listo para análisis
        """
        # Handle None or non-string input
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        
        # Lemmatize and filter
        cleaned_tokens = []
        for token in tokens:
            if len(token) > 2 and token not in self.stop_words:
                try:
                    lemma = self.lemmatizer.lemmatize(token)
                except Exception:
                    lemma = token
                if lemma not in self.stop_words and len(lemma) > 2:
                    cleaned_tokens.append(lemma)
        
        return ' '.join(cleaned_tokens)
    
    def fit_transform(self, comments: List[str]) -> Dict:
        """
        Entrena el modelo LDA y extrae los temas principales.
        
        Args:
            comments: Lista de textos/comentarios a analizar
            
        Returns:
            Diccionario con los topics identificados y sus palabras clave
            
        Raises:
            ValueError: Si la lista de comentarios está vacía o es inválida
        """
        # Validate input
        if comments is None:
            return {"error": "Comments cannot be None"}
        
        if not isinstance(comments, (list, tuple)):
            return {"error": "Comments must be a list or tuple"}
        
        if len(comments) == 0:
            return {"error": "No comments provided"}
        
        # Preprocess all comments
        processed_comments = [self.preprocess_text(c) for c in comments]
        
        # Filter empty comments
        processed_comments = [c for c in processed_comments if c.strip()]
        
        if len(processed_comments) < 10:
            return {"error": "Not enough valid comments after preprocessing"}
        
        # Create document-term matrix
        try:
            doc_term_matrix = self.vectorizer.fit_transform(processed_comments)
        except ValueError:
            return {"error": "Not enough vocabulary in comments"}
        
        # Get feature names
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Fit LDA
        self.lda_model.fit(doc_term_matrix)
        self.fitted = True
        
        # Extract topics
        topics = self._extract_topics()
        
        # Get document-topic distribution
        doc_topics = self.lda_model.transform(doc_term_matrix)
        
        # Calculate topic distribution per document
        topic_distribution = doc_topics.mean(axis=0)
        
        return {
            "topics": topics,
            "topic_distribution": topic_distribution.tolist(),
            "n_comments_analyzed": len(processed_comments),
            "n_topics": self.n_topics,
            "doc_topic_matrix": doc_topics
        }
    
    def _extract_topics(self) -> List[Dict]:
        """Extrae los temas principales"""
        topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            # Get top words for this topic
            top_word_indices = topic.argsort()[:-self.n_top_words - 1:-1]
            top_words = [self.feature_names[i] for i in top_word_indices]
            top_weights = [topic[i] for i in top_word_indices]
            
            # Normalize weights
            total_weight = sum(top_weights)
            top_weights = [w / total_weight for w in top_weights]
            
            # Calculate topic percentage
            topic_percentage = self.lda_model.components_[topic_idx].sum() / self.lda_model.components_.sum()
            
            topics.append({
                "topic_id": topic_idx,
                "topic_words": top_words,
                "topic_weights": top_weights,
                "topic_percentage": round(topic_percentage * 100, 2),
                "keywords": top_words[:5]  # Top 5 keywords
            })
        
        # Sort by percentage
        topics.sort(key=lambda x: x["topic_percentage"], reverse=True)
        
        return topics
    
    def get_topic_for_document(self, text: str) -> Dict:
        """Obtiene el tema principal de un documento/comentario"""
        if not self.fitted:
            return {"error": "Model not fitted"}
        
        processed = self.preprocess_text(text)
        doc_vector = self.vectorizer.transform([processed])
        topic_probs = self.lda_model.transform(doc_vector)[0]
        
        dominant_topic = int(np.argmax(topic_probs))
        
        return {
            "dominant_topic": dominant_topic,
            "topic_probabilities": topic_probs.tolist(),
            "topic_name": f"Topic {dominant_topic}"
        }
    
    def label_topics(self, topics: List[Dict]) -> List[Dict]:
        """Intenta etiquetar los temas basándose en palabras clave"""
        
        topic_keywords = {
            "habilidad_técnica": ["skill", "jugar", "mecanica", "aim", "posición", "carry", "moves", "gameplay", "technique"],
            "frecuencia_streams": ["stream", "cuando", "vuelve", "ausente", "seguido", "frecuente", "falta", "esperar", "nunca"],
            "entretenimiento": ["gracia", "divertido", "aburrido", "entretenido", "reir", "funny", "entertaining", "boring", "lol"],
            "comunidad": ["comunidad", "fan", "apoyo", "gracias", "community", "fan", "support", "thanks"],
            "colaboraciones": ["collab", "colabora", "amigo", "ft", "featuring", "with", "together"],
            "contenido": ["video", "subir", "contenido", "youtube", "canal", "content", "upload"],
            "personalidad": ["persona", "carácter", "attitude", "personality", "attractive", "charism"],
            "crítica": ["malo", "peor", "terrible", "bad", "worse", "hate", "worst", "terrible"]
        }
        
        labeled_topics = []
        
        for topic in topics:
            topic_words = topic.get("topic_words", [])
            topic_text = " ".join(topic_words).lower()
            
            # Find matching label
            best_label = "tema_general"
            best_match = 0
            
            for label, keywords in topic_keywords.items():
                matches = sum(1 for kw in keywords if kw in topic_text)
                if matches > best_match:
                    best_match = matches
                    best_label = label
            
            labeled_topics.append({
                **topic,
                "topic_name": best_label
            })
        
        return labeled_topics
    
    def analyze_channel(self, comments: List[Dict]) -> Dict:
        """Analiza comentarios de un canal y genera reporte"""
        
        # Extract text from comments
        texts = [c.get("text", "") for c in comments]
        
        # Run LDA
        result = self.fit_transform(texts)
        
        if "error" in result:
            return result
        
        # Label topics
        labeled_topics = self.label_topics(result["topics"])
        
        # Generate summary
        summary = []
        for topic in labeled_topics:
            summary.append(
                f"- {topic['topic_percentage']:.1f}% de los comentarios hablan sobre "
                f"'{topic['topic_name']}' (palabras clave: {', '.join(topic['keywords'])})"
            )
        
        return {
            "n_topics": len(labeled_topics),
            "topics": labeled_topics,
            "n_comments_analyzed": result["n_comments_analyzed"],
            "summary": "\n".join(summary)
        }
