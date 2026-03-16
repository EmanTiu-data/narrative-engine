"""
Correlation Engine
Calcula correlaciones entre plataformas usando Pearson y Lag Analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta


class CorrelationEngine:
    """Motor de correlación multi-plataforma"""
    
    def __init__(self, min_correlation: float = 0.3, significance_level: float = 0.05):
        self.min_correlation = min_correlation
        self.significance_level = significance_level
    
    def normalize_series(self, series: pd.Series, method: str = "minmax") -> pd.Series:
        """Normaliza una serie para que sea comparable entre plataformas"""
        
        # Handle NaN
        series = series.fillna(0)
        
        if method == "minmax":
            # Min-Max normalization: (x - min) / (max - min)
            min_val = series.min()
            max_val = series.max()
            if max_val - min_val == 0:
                return pd.Series([0.5] * len(series), index=series.index)
            return (series - min_val) / (max_val - min_val)
        
        elif method == "zscore":
            # Z-Score normalization: (x - mean) / std
            mean = series.mean()
            std = series.std()
            if std == 0:
                return pd.Series([0] * len(series), index=series.index)
            return (series - mean) / std
        
        elif method == "percentile":
            # Percentile ranking: convert to percentiles 0-100
            return series.rank(pct=True) * 100
        
        return series
    
    def normalize_for_correlation(self, series_a: pd.Series, series_b: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Normaliza ambas series antes de calcular correlación"""
        
        # Align indices
        common_idx = series_a.index.intersection(series_b.index)
        s_a = series_a.loc[common_idx]
        s_b = series_b.loc[common_idx]
        
        # Apply Min-Max normalization to both
        norm_a = self.normalize_series(s_a, "minmax")
        norm_b = self.normalize_series(s_b, "minmax")
        
        return norm_a, norm_b
    
    def calculate_pearson(self, series_a: pd.Series, series_b: pd.Series, 
                         normalize: bool = True) -> Dict:
        """Calcula coeficiente de correlación de Pearson"""
        
        # Remove NaN values
        mask = ~(series_a.isna() | series_b.isna())
        x = series_a[mask]
        y = series_b[mask]
        
        if len(x) < 3:
            return {
                "r": 0,
                "p_value": 1,
                "significant": False,
                "n_observations": len(x)
            }
        
        # Calculate Pearson correlation
        r, p_value = stats.pearsonr(x, y)
        
        return {
            "r": round(r, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < self.significance_level,
            "n_observations": len(x),
            "correlation_strength": self._interpret_strength(r)
        }
    
    def _interpret_strength(self, r: float) -> str:
        """Interpreta la fuerza de la correlación"""
        abs_r = abs(r)
        
        if abs_r >= 0.7:
            return "strong"
        elif abs_r >= 0.4:
            return "moderate"
        elif abs_r >= 0.2:
            return "weak"
        else:
            return "negligible"
    
    def calculate_lag_correlation(self, series_a: pd.Series, series_b: pd.Series, 
                                  max_lag: int = 14, normalize: bool = True) -> Dict:
        """Calcula correlación con diferentes lags (retrasos)"""
        
        # Normalize series before correlation
        if normalize and len(series_a) > 0 and len(series_b) > 0:
            series_a, series_b = self.normalize_for_correlation(series_a, series_b)
        
        results = []
        
        for lag in range(-max_lag, max_lag + 1):
            # Shift series_b by lag
            if lag > 0:
                shifted = series_b.shift(lag)
            elif lag < 0:
                shifted = series_b.shift(lag)
            else:
                shifted = series_b
            
            # Calculate correlation (with normalization disabled since we already did it)
            corr_result = self.calculate_pearson(series_a, shifted, normalize=False)
            
            results.append({
                "lag": lag,
                "r": corr_result["r"],
                "p_value": corr_result["p_value"],
                "significant": corr_result["significant"]
            })
        
        # Find optimal lag
        significant_results = [r for r in results if r["significant"]]
        
        if significant_results:
            # Find the one with highest absolute correlation
            best = max(significant_results, key=lambda x: abs(x["r"]))
            optimal_lag = best["lag"]
            best_r = best["r"]
        else:
            # If none significant, find highest correlation anyway
            best = max(results, key=lambda x: abs(x["r"]))
            optimal_lag = best["lag"]
            best_r = best["r"]
        
        return {
            "optimal_lag": optimal_lag,
            "best_correlation": round(best_r, 4),
            "lag_results": results,
            "has_significant_correlation": len(significant_results) > 0,
            "interpretation": self._interpret_lag(optimal_lag, best_r)
        }
    
    def _interpret_lag(self, lag: int, r: float) -> str:
        """Interpreta el resultado del lag"""
        if lag == 0:
            lag_desc = "mismo día"
        elif lag > 0:
            lag_desc = f"+{lag} días"
        else:
            lag_desc = f"{lag} días"
        
        if abs(r) >= 0.7:
            strength = "fuerte"
        elif abs(r) >= 0.4:
            strength = "moderada"
        else:
            strength = "débil"
        
        direction = "positiva" if r > 0 else "negativa"
        
        return f"Correlación {strength} {direction} con lag de {lag_desc}"
    
    def align_time_series(self, *series_list: pd.Series) -> pd.DataFrame:
        """Alinea múltiples series temporales"""
        
        # Combine all series
        df = pd.concat(series_list, axis=1)
        
        # Forward fill missing values
        df = df.ffill()
        
        # Backward fill for any remaining NaNs
        df = df.bfill()
        
        return df
    
    def calculate_platform_correlation(self, df_a: pd.DataFrame, metric_a: str,
                                      df_b: pd.DataFrame, metric_b: str,
                                      date_col_a: str = "date", date_col_b: str = "date") -> Dict:
        """Calcula correlación entre dos plataformas"""
        
        # Ensure date columns are datetime
        df_a = df_a.copy()
        df_b = df_b.copy()
        
        if date_col_a in df_a.columns:
            df_a[date_col_a] = pd.to_datetime(df_a[date_col_a])
        if date_col_b in df_b.columns:
            df_b[date_col_b] = pd.to_datetime(df_b[date_col_b])
        
        # Merge on date
        merged = pd.merge(
            df_a[[date_col_a, metric_a]],
            df_b[[date_col_b, metric_b]],
            left_on=date_col_a,
            right_on=date_col_b,
            how="inner"
        )
        
        if len(merged) < 3:
            return {
                "error": "No enough overlapping data points",
                "n_points": len(merged)
            }
        
        # Calculate correlation
        pearson = self.calculate_pearson(merged[metric_a], merged[metric_b])
        
        # Calculate lag correlation
        lag_result = self.calculate_lag_correlation(
            merged[metric_a], 
            merged[metric_b],
            max_lag=7
        )
        
        return {
            "pearson": pearson,
            "lag_analysis": lag_result,
            "n_data_points": len(merged),
            "date_range": {
                "start": str(merged[date_col_a].min()),
                "end": str(merged[date_col_a].max())
            }
        }
    
    def calculate_all_correlations(self, platform_data: Dict[str, pd.DataFrame]) -> Dict:
        """Calcula correlaciones entre todas las plataformas disponibles"""
        
        correlations = {}
        
        # Get all platform pairs
        platforms = list(platform_data.keys())
        
        for i, platform_a in enumerate(platforms):
            for platform_b in platforms[i + 1:]:
                key = f"{platform_a}_vs_{platform_b}"
                
                try:
                    # Try common metrics
                    result = self._try_correlation(
                        platform_data[platform_a],
                        platform_data[platform_b]
                    )
                    
                    correlations[key] = result
                    
                except Exception as e:
                    correlations[key] = {
                        "error": str(e),
                        "platform_a": platform_a,
                        "platform_b": platform_b
                    }
        
        return correlations
    
    def _try_correlation(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict:
        """Intenta calcular correlación con métricas comunes"""
        
        # Define common metrics to try
        metrics_a = ["views", "followers", "streams", "total_views", "avg_viewers"]
        metrics_b = ["views", "followers", "streams", "total_views", "avg_viewers"]
        
        # Try each combination
        best_result = {"error": "No se encontraron métricas compatibles"}
        
        for metric_a in metrics_a:
            for metric_b in metrics_b:
                if metric_a in df_a.columns and metric_b in df_b.columns:
                    try:
                        result = self.calculate_platform_correlation(
                            df_a, metric_a,
                            df_b, metric_b
                        )
                        
                        if "error" not in result and result.get("pearson", {}).get("significant"):
                            if "error" in best_result or abs(result["pearson"]["r"]) > abs(best_result.get("pearson", {}).get("r", 0)):
                                best_result = result
                                best_result["metric_a"] = metric_a
                                best_result["metric_b"] = metric_b
                    
                    except:
                        continue
        
        return best_result


class GracefulDegradation:
    """Maneja la degradação elegante cuando una plataforma no está disponible"""
    
    def __init__(self):
        self.correlation_engine = CorrelationEngine()
    
    def calculate_available_correlations(self, platform_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calcula correlaciones solo con las plataformas disponibles.
        Si una plataforma falla, se ignora y se calculan correlaciones con las demás.
        """
        
        available_platforms = {}
        unavailable_platforms = []
        
        # Check which platforms have data
        for platform, df in platform_data.items():
            if df is not None and len(df) > 0:
                available_platforms[platform] = df
            else:
                unavailable_platforms.append(platform)
        
        if len(available_platforms) < 2:
            return {
                "status": "insufficient_data",
                "available_platforms": list(available_platforms.keys()),
                "unavailable_platforms": unavailable_platforms,
                "correlations": {},
                "message": "Se necesitan al menos 2 plataformas con datos"
            }
        
        # Calculate correlations for available platforms
        correlations = self.correlation_engine.calculate_all_correlations(available_platforms)
        
        # Filter significant correlations
        significant = {}
        for key, result in correlations.items():
            if "error" not in result:
                if result.get("pearson", {}).get("significant"):
                    significant[key] = result
        
        return {
            "status": "success",
            "available_platforms": list(available_platforms.keys()),
            "unavailable_platforms": unavailable_platforms,
            "total_correlations": len(correlations),
            "significant_correlations": significant,
            "all_correlations": correlations,
            "message": f"Calculadas correlaciones para {len(available_platforms)} plataformas"
        }
    
    def generate_report(self, result: Dict) -> str:
        """Genera un reporte textual de las correlaciones"""
        
        if result["status"] == "insufficient_data":
            return f"""
⚠️ Datos insuficientes para correlación
Plataformas disponibles: {', '.join(result.get('available_platforms', []))}
Plataformas no disponibles: {', '.join(result.get('unavailable_platforms', []))}
            """
        
        lines = [
            "=" * 60,
            "REPORTE DE CORRELACIONES ENTRE PLATAFORMAS",
            "=" * 60,
            f"Plataformas analizadas: {', '.join(result['available_platforms'])}",
            f"Correlaciones significativas: {len(result.get('significant_correlations', {}))}",
            ""
        ]
        
        if result.get("significant_correlations"):
            lines.append("📊 CORRELACIONES SIGNIFICATIVAS:")
            lines.append("-" * 40)
            
            for key, corr in result["significant_correlations"].items():
                r = corr.get("pearson", {}).get("r", 0)
                p = corr.get("pearson", {}).get("p_value", 1)
                lag = corr.get("lag_analysis", {}).get("optimal_lag", 0)
                interp = corr.get("lag_analysis", {}).get("interpretation", "")
                
                lines.append(f"\n{key}:")
                lines.append(f"  • Correlación: r = {r:.3f} (p = {p:.4f})")
                lines.append(f"  • Lag óptimo: {lag} días")
                lines.append(f"  • {interp}")
        
        if result.get("unavailable_platforms"):
            lines.append(f"\n⚠️ Plataformas no disponibles: {', '.join(result['unavailable_platforms'])}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
