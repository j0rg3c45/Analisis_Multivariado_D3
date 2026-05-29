# Situacion 4: Analisis Multitemporal de Cobertura Forestal (Landsat)

**Dataset:** Landsat Science (USGS) - Cubo Espaciotemporal UAO  
**Indices disponibles:** NDVI, NBR, EVI, NDMI (4 indices x 11 anos = 44 variables)  
**Periodo:** 2015-2025  
**Escala:** Validada en tiempo de ejecucion. Si los valores del CSV son DN enteros
(max > 1.5), el script aplica automaticamente la formula Landsat Collection 2:
`reflectancia = valor * 0.0000275 - 0.2`. Si ya estan en escala normalizada (-1 a 1),
se usan directamente. El script reporta en consola cual ruta tomo para cada indice.  
**Objetivo:** Identificar, aislar y caracterizar estructuras latentes que expliquen patrones de cambio en cobertura forestal usando ACP, ICA, IVA y clustering.

---

## 1. Construccion de Cubo Multivariado Espacio-Temporal

- Integrar serie temporal de imagenes Landsat (Landsat 5, 7, 8, 9).
- Periodo: 11 anos (2015-2025).
- Extraer bandas espectrales relevantes:
  - Rojo (para NDVI, EVI)
  - Infrarrojo Cercano - NIR (para NDVI, EVI, NDMI)
  - Infrarrojo de Onda Corta - SWIR (para NBR, NDMI)
- Calcular indices de vegetacion clave para cada imagen:
  - **NDVI** (Normalized Difference Vegetation Index) - salud vegetativa general
  - **EVI** (Enhanced Vegetation Index) - corrige influencia atmosferica y de suelo
  - **NBR** (Normalized Burn Ratio) - severidad de incendios / biomasa
  - **NDMI** (Normalized Difference Moisture Index) - contenido de humedad
- Correccion de escala: validada en runtime; se aplica factor 0.0000275 - 0.2 solo si
  los valores del CSV son DN enteros (Collection 2 Landsat); si son indices ya en
  escala normalizada, se omite la correccion para evitar doble transformacion.
- Generar dataset multivariado (cubo) por pixel:
  - Observaciones: cada pixel
  - Variables: indices de cada ano (44 variables)

## 2. Identificacion de Componentes Latentes (ACP / ICA)

### PCA (o Sparse PCA)
- Aplicar al cubo de datos para capturar variabilidad temporal dominante.
- Identificar principales modos de cambio.
- Evaluar varianza explicada acumulada.

### ICA
- Separar fuentes independientes de cambio:
  - Perturbaciones abruptas (incendios)
  - Degradacion progresiva
  - Recuperacion post-evento
- Interpretar componentes vinculandolas con eventos ecologicos plausibles (cambio de uso, sequias).

## 3. Aplicacion Obligatoria de IVA

- Tratar diferentes fuentes de variacion como "vistas" para modelo IVA.
- Vistas exploradas:
  - **Par 1:** NDVI <-> NBR (salud vegetativa vs biomasa/quemas)
  - **Par 2:** EVI <-> NDMI (vigor corregido vs humedad)
  - **Par 3:** NDVI <-> EVI (comparacion de indices verdes)
- Objetivo: Identificar componentes independientes multivista que revelen patrones persistentes de deforestacion, regeneracion o degradacion capturados simultaneamente por diferentes indices.

## 4. Deteccion de Zonas de Cambio (Clustering)

- Sobre factores/componentes extraidos (IVA):
  - **K-means**
  - **GMM** (Gaussian Mixture Models)
- Evaluacion comparativa: Silhouette Score, Davies-Bouldin, Calinski-Harabasz
- Segmentar pixeles en regiones con trayectorias de cambio homogeneas:
  - Perdida significativa de cobertura
  - Estabilidad ecologica
  - Recuperacion forestal activa
  - Patrones transicionales
- Generar mapas tematicos con distribucion espacial de clusters.
- Generar trayectorias NDVI con bandas de desviacion estandar (±1σ) por cluster.

## 5. Evaluacion Cuantitativa del Cambio

Para cada cluster identificado, calcular por cada indice:
- Tasas de deforestacion/recuperacion anual
- Magnitud del cambio en indices
- Lineas de tendencia temporal
- Incluir visualizacion multivariada y mapas de cambio.

## 6. Interpretacion Ecologica y Conclusiones

- Interpretar factores latentes y clusters en terminos de procesos:
  - Ecologicos (regeneracion natural, sucesion)
  - Antropogenicos (expansion agricola, areas protegidas)
- Discutir limitaciones:
  - Correccion de escala: verificada en runtime; puede no aplicarse si el CSV
    ya contiene indices en escala normalizada
  - Cobertura de nubes en datos Landsat (valores faltantes imputados con mediana)
  - Resolucion espacial/temporal
- Aplicaciones de hallazgos en:
  - Monitoreo forestal
  - Conservacion
  - Politicas ambientales

## Entregables

- Codigo Python con implementacion completa (`scripts_desafio/situacion_04.py`).
- Mapas tematicos de clusters de cambio (`situacion_04_mapa_clusters.png`).
- Tablas y graficos de tendencias temporales multi-indice:
  - `situacion_04_trayectorias_indices.png` (4 indices × 4 clusters)
  - `situacion_04_trayectorias_ndvi.png` (NDVI con bandas ±1σ, tasa anual en leyenda)
  - `situacion_04_cambio_indices.png` (cambio total y tasa anual)
- Mapas de componentes latentes (`situacion_04_mapas_latentes.png`).
- Interpretacion ecologica de resultados con consistencia multi-indice validada.
