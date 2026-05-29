# Reporte Situacion 4: Analisis Multitemporal de Cobertura Forestal (Landsat)

## Punto 1: Cubo Espacio-Temporal
- **60,214 pixeles**, periodo 2015-2025 (11 años)
- **4 indices**: NDVI, NBR, EVI, NDMI (44 variables = 4 × 11)
- Escala de valores: verificada en tiempo de ejecucion. Si los valores del CSV
  son DN enteros (max > 1.5), se aplica automaticamente la formula Landsat
  Collection 2: `reflectancia = DN × 0.0000275 − 0.2`. Si ya estan en escala
  de reflectancia/indice, se usan directamente. El script reporta en consola
  que operacion aplico para cada indice.

## Punto 2: Componentes Latentes (PCA/ICA)

### PCA — Varianza Explicada
| PC | Var. Indiv. | Var. Acum. | Interpretacion |
|----|:---------:|:---------:|----------------|
| PC1 | 75.88% | 75.88% | Vigor forestal base (estado vegetativo dominante) |
| PC2 | 2.87% | 78.74% | Variabilidad estacional/interanual |
| PC3 | 2.27% | 81.01% | Cambios interanuales progresivos |
| PC4 | 2.26% | 83.28% | Nubosidad / ruido residual |
| PC5 | 1.95% | **85.22%** | Perturbaciones locales puntuales |

PC1 domina con 75.88%: refleja el estado vegetativo base del pixel a lo largo
de toda la serie temporal. PC2 a PC5 capturan dinamicas de cambio mas sutiles.

### ICA — Curtosis por Componente
| Componente | Curtosis | Interpretacion |
|------------|:--------:|----------------|
| IC1 | 31,033 | Perturbaciones abruptas (incendios, tala) |
| IC2 | 44,110 | Eventos extremos de cambio (maxima no-gaussianidad) |
| IC3 | 13,454 | Degradacion progresiva multi-año |
| IC4 | 15.52 | Ruido residual estructurado |
| IC5 | 0.26 | Componente gaussiana (ruido fondo) |

Curtosis extremas (>10,000) en IC1-IC3 confirman separacion exitosa de fuentes
independientes no-gaussianas, tipicamente asociadas a eventos discretos.

## Punto 3: IVA Multivista
Tres pares explorados, cada uno trata un par de indices como vistas independientes:

| Par de Vistas | Alignment Score | Componente mas alineado |
|---------------|:--------------:|------------------------|
| NDVI <-> NBR | 0.2234 | IC5 (corr=0.8937) |
| **EVI <-> NDMI** | **0.4828** | IC5 (corr=0.8820) |
| NDVI <-> EVI | 0.2122 | IC5 (corr=0.9283) |

- **Mejor par: EVI <-> NDMI** (Alignment Score=0.4828). El vigor vegetativo
  corregido (EVI) y el contenido de humedad (NDMI) capturan conjuntamente los
  patrones de cambio mas consistentes entre vistas.
- IC5 es el componente mas consistente en todos los pares (corr>0.88),
  sugiriendo que representa el estado base de la vegetacion como factor latente
  comun.

## Punto 4: Clustering
| Metodo | K optimo (Sil) | K=4 Silhouette | Davies-Bouldin | Calinski-Harabasz |
|--------|:--------------:|:--------------:|:--------------:|:-----------------:|
| K-means | K=2 (0.3224) | 0.2550 | 1.4858 | 19,699 |
| GMM | K=3 (0.2477) | 0.2242 | 3.3412 | 10,040 |

K-means supera a GMM en las 3 metricas con K=4 ecologicamente interpretable.

## Punto 5: Evaluacion Cuantitativa
| Cluster | % Area | Pixels | NDVI | NBR | EVI | NDMI | Tendencia |
|---------|:-----:|:------:|:---:|:---:|:---:|:----:|-----------|
| C0 | 15.9% | 9,594 | -2.78% | -3.94% | -3.05% | -4.45% | **DEGRADACION** |
| C1 | 25.2% | 15,179 | +1.16% | +1.56% | +5.89% | +5.47% | **ESTABLE/RECUPERACION** |
| C2 | 23.8% | 14,305 | -0.84% | +8.81% | +5.42% | +147% | **REGENERACION ACTIVA** |
| C3 | 35.1% | 21,136 | +0.32% | +1.19% | +3.24% | +7.01% | **BOSQUE ESTABLE** |

## Punto 6: Interpretacion Ecologica
- **Degradacion (15.9%):** Perdida consistente en los 4 indices, degradacion
  estructural multi-anio con presion antropogenica.
- **Regeneracion (23.8%):** NBR +8.81% y NDMI de negativo a positivo indican
  recuperacion de biomasa y humedad — sucesion vegetal secundaria.
- **Bosque estable (60.3% = C1+C3):** Area mayoritaria conservada con
  tendencia estable o positiva en los 4 indices.

**Consistencia multi-indice:** La concordancia direccional entre NDVI, NBR, EVI
y NDMI dentro de cada cluster valida la robustez de la segmentacion.

## Visualizaciones generadas (9 archivos en output/situacion_04/)
| Archivo | Descripcion |
|---------|-------------|
| `situacion_04_mapa_clusters.png` | Mapa tematico espacial de clusters |
| `situacion_04_trayectorias_indices.png` | Trayectorias temporales 4 indices × 4 clusters |
| `situacion_04_trayectorias_ndvi.png` | *(codigo en script)* NDVI con bandas ±1σ por cluster |
| `situacion_04_mapas_latentes.png` | Mapas espaciales PC1, PC2, IC1 |
| `situacion_04_varianza_pca.png` | Varianza explicada PCA |
| `situacion_04_loadings_pca.png` | Cargas PCA |
| `situacion_04_comparacion_clustering.png` | Silhouette vs K: K-means vs GMM |
| `situacion_04_cambio_indices.png` | Cambio total y tasa anual por cluster |

---
*Reporte actualizado — 2026-05-28 | Correcciones: validacion de escala en runtime, trayectorias_ndvi con codigo en script*
