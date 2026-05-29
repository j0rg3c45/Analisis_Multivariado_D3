# Reporte Situacion 4: Analisis Multitemporal de Cobertura Forestal (Landsat)

**Dataset:** Landsat_Cubo_Espaciotemporal_UAO.csv — 60,214 píxeles | 2015–2025  
**Ejecución:** 2026-05-28 | Datos reales confirmados

## Punto 1: Cubo Espacio-Temporal

- **60,214 píxeles**, periodo 2015–2025 (11 años)
- **4 índices:** NDVI, NBR, EVI, NDMI → **44 variables** (4 × 11)
- Escala validada en runtime:

| Índice | Resultado |
|--------|-----------|
| EVI  | Corrección Landsat C2 aplicada (DN→reflectancia) |
| NDVI | Corrección Landsat C2 aplicada (DN→reflectancia) |
| NBR  | Valores ya en escala de reflectancia/índice |
| NDMI | Valores ya en escala de reflectancia/índice |

El CSV contiene formatos mixtos (EVI y NDVI en DN enteros; NBR y NDMI ya normalizados). El script los detecta y corrige automáticamente.

## Punto 2: Componentes Latentes (PCA/ICA)

### PCA — Varianza Explicada

| PC | Var. Indiv. | Var. Acum. | Interpretación |
|----|:---------:|:---------:|----------------|
| PC1 | 75.88% | 75.88% | Vigor forestal base (estado vegetativo dominante) |
| PC2 | 2.87% | 78.74% | Variabilidad estacional/interanual |
| PC3 | 2.27% | 81.01% | Cambios interanuales progresivos |
| PC4 | 2.26% | 83.28% | Nubosidad / ruido residual |
| PC5 | 1.95% | **85.22%** | Perturbaciones locales puntuales |

PC1 domina con 75.88%: refleja el estado vegetativo base del píxel en toda la serie temporal. La caída abrupta de PC2 en adelante (<3%) muestra que el cambio temporal es un fenómeno minoritario frente al estado base.

### ICA — Curtosis por Componente

| Componente | Curtosis | Interpretación |
|------------|:--------:|----------------|
| IC1 | 31,033 | Perturbaciones abruptas (incendios, tala) |
| IC2 | 44,110 | Eventos extremos de cambio (máxima no-gaussianidad) |
| IC3 | 13,454 | Degradación progresiva multi-año |
| IC4 | 15.52 | Ruido residual estructurado |
| IC5 | 0.26 | Componente gaussiana (ruido de fondo) |

Curtosis extremas (>10,000) en IC1–IC3 confirman separación exitosa de fuentes independientes no-gaussianas, típicamente asociadas a eventos discretos de disturbio forestal.

## Punto 3: IVA Multivista

| Par de Vistas | Alignment Score | Componente más alineado |
|---------------|:--------------:|------------------------|
| NDVI ↔ NBR | 0.2234 | IC5 (corr=0.8937) |
| **EVI ↔ NDMI** | **0.4828** | IC5 (corr=0.8820) |
| NDVI ↔ EVI | 0.2122 | IC5 (corr=0.9283) |

**Mejor par: EVI ↔ NDMI** (Alignment Score=0.4828). EVI (vigor corregido) y NDMI (humedad) responden a los mismos procesos ecológicos, produciendo la mayor dependencia multivista. IC5 es el componente más consistente en los 3 pares (corr>0.88): representa el estado base de la vegetación como factor latente común.

## Punto 4: Clustering

| Método | K óptimo (Sil) | K=4 Silhouette | Davies-Bouldin | Calinski-Harabasz |
|--------|:--------------:|:--------------:|:--------------:|:-----------------:|
| K-means | K=2 (0.3224) | 0.2550 | 1.4858 | 19,699 |
| GMM | K=3 (0.2477) | 0.2242 | 3.3412 | 10,040 |

K-means seleccionado (superior en las 3 métricas con K=4 ecológicamente interpretable).

## Punto 5: Evaluación Cuantitativa del Cambio

| Cluster | % Área | Píxeles | ΔNDVI | ΔNBR | ΔEVI | ΔNDMI | Clasificación |
|---------|:-----:|:------:|:----:|:----:|:----:|:-----:|-----------|
| C0 | 15.9% | 9,594 | -1.00% | -2.10% | -4.26% | -3.39% | **DEGRADACIÓN** |
| C1 | 25.2% | 15,179 | +4.29% | +6.17% | +5.86% | +11.97% | **REGENERACIÓN ACTIVA** |
| C2 | 23.8% | 14,305 | +5.34% | +26.80% | +9.76% | +240%* | **REGENERACIÓN ACTIVA** |
| C3 | 35.1% | 21,136 | +4.16% | +6.59% | +4.99% | +13.67% | **REGENERACIÓN ACTIVA** |

(*) NDMI C2: cambio porcentual inflado por denominador ≈0 (inicio: -0.009, fin: +0.013 — cambio absoluto relevante).

## Punto 6: Interpretación Ecológica

- **Degradación (C0 — 15.9%):** Pérdida consistente en los 4 índices. EVI −4.26% y NDMI −3.39% indican pérdida de vigor y humedad del dosel. Concentración geográfica sugiere presión antropogénica (tala selectiva, expansión agrícola).

- **Regeneración incipiente (C2 — 23.8%):** Zona de baja cobertura inicial (NDVI≈0.30). NBR +26.80% y NDMI de negativo a positivo: recuperación activa de biomasa y contenido hídrico. Típico de sucesión vegetal secundaria post-deforestación.

- **Bosque en recuperación activa (C1+C3 — 60.3%):** El grueso del área (NDVI≈0.68–0.71) muestra tendencias positivas en todos los índices. NDMI es el más sensible (+11.97% en C1, +13.67% en C3): refleja recuperación del contenido hídrico del dosel antes que el verdor sea visible en NDVI.

**Consistencia multi-índice:** La concordancia direccional entre NDVI, NBR, EVI y NDMI dentro de cada clúster valida la robustez de la segmentación.

## Visualizaciones (8 archivos en `output/situacion_04/`)

| Archivo | Descripción |
|---------|-------------|
| `situacion_04_mapa_clusters.png` | Mapa temático espacial de clústeres |
| `situacion_04_trayectorias_indices.png` | Trayectorias temporales 4 índices × 4 clústeres |
| `situacion_04_trayectorias_ndvi.png` | NDVI con bandas ±1σ por clúster (tasa anual en leyenda) |
| `situacion_04_mapas_latentes.png` | Mapas espaciales PC1, PC2, IC1 |
| `situacion_04_varianza_pca.png` | Varianza explicada PCA |
| `situacion_04_comparacion_clustering.png` | Silhouette vs K: K-means vs GMM |
| `situacion_04_cambio_indices.png` | Cambio total y tasa anual por clúster |

---
*Reporte final con métricas reales — 2026-05-28 | Dataset: 60,214 píxeles | 2015–2025*
