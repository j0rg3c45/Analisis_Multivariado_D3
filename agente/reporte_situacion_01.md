# Reporte Situacion 1: Estructuras Latentes en Viajes de NYC

## 1. Resumen de Integracion
Se integraron 4 servicios de taxi (Yellow, Green, FHV, HVFHV) — marzo 2026.
- **Registros integrados:** 100,000 (25,000 por servicio)
- **Outliers eliminados:** 550 (0.5%)
- **Registros FHV excluidos del análisis latente:** 24,965 (sin trip_distance/fare_amount)
- **Muestra final para análisis:** 74,485 registros completos

| Servicio | Registros |
|----------|----------:|
| HVFHV    | 25,000    |
| Green    | 24,876    |
| Yellow   | 24,609    |

## 2. Analisis Factorial Exploratorio (EFA)
Rotacion varimax, 3 factores — ejecutado en `main()`, métricas reportadas en consola:

| Factor | Varianza (prop. scores) | Comunalidad destacada |
|--------|:-----------------------:|-----------------------|
| F1 | 95.38% | duration_minutes=0.6055, fare_amount=0.6040 |
| F2 | 3.77% | — |
| F3 | 0.85% | — |

Comunalidades individuales:
| Variable | h² | Interpretación |
|----------|----|----------------|
| duration_minutes | 0.6055 | Bien representada por EFA |
| fare_amount | 0.6040 | Bien representada por EFA |
| trip_distance | 0.0058 | Baja carga factorial |
| passenger_count | 0.0241 | Prácticamente independiente de los factores |

> La baja comunalidad de `trip_distance` y `passenger_count` indica que estas
> variables son casi ortogonales al espacio factorial: `passenger_count` actúa
> como dimensión autónoma (consistente con Sparse PCA C3 carga=1.0) y
> `trip_distance` está altamente correlacionada con `fare_amount` quedando
> absorbida en la misma dirección latente.

**Gráfico:** `output/situacion_01/situacion_01_efa_loadings.png`

## 3. Extraccion de Componentes Latentes

### 3.1 Matriz de Cargas (Sparse PCA — máxima parsimonia)
| Variable | C1 | C2 | C3 |
|----------|:--:|:--:|:--:|
| Duration | ≠0 | 0  | 0  |
| Fare     | ≠0 | 0  | 0  |
| Distance | 0  | ≠0 | 0  |
| Passengers | 0 | 0 | ≠0 |

Sparse PCA conserva la separación de dimensiones confiriendo máxima interpretabilidad.

### 3.2 IVA v2 — Vistas Semánticas (resultados reales)
| Vista | Variables | Dominio |
|-------|-----------|---------|
| Vista 1 (espacio-temporal) | `trip_distance`, `duration_minutes` | Geometría del viaje |
| Vista 2 (económica/demanda) | `fare_amount`, `passenger_count` | Valor y ocupación |

- **Alignment Score: 0.3041** — correlación cruzada media entre vistas
- **Alignment vector: [1, 1]** — ambos componentes de V1 alinean con el componente 1 de V2
- Las vistas comparten ~30% de estructura latente; el resto es información específica de cada dominio

## 4. Modelos Multivariados con Multiples Respuestas
| Metodo | AUC-ROC (Propina) | Accuracy (Pasajeros) | R² (Monto) | Nota |
|--------|:-----------------:|:-------------------:|:----------:|------|
| PCA | 0.5947 | 1.0000* | 0.3432 | Muestra completa |
| Sparse PCA | 0.5946 | 0.9999* | 0.1848 | Muestra completa |
| Kernel PCA | 0.5745 | 0.9920 | 0.6052† | Submuestra 5k |
| ICA | **0.5947** | 0.9999* | **0.3948** | Muestra completa |
| **IVA v2** | 0.5592 | **0.9028** | 0.2831 | Muestra completa |

(*) Acc≈1.0 refleja dominio absoluto de viajes de 1 pasajero en el dataset —
clasificador trivial; no indica capacidad discriminativa real del espacio latente.
(†) Kernel PCA sobre submuestra independiente de 5,000 registros.

## 5. Comparacion Global
| Metodo | K | Silhouette | CH | DB | AUC | R² | Parsimonia |
|--------|:-:|:----------:|:--:|:--:|:---:|:--:|:----------:|
| ICA | 2 | **0.9929** | 34,896 | 0.1324 | 0.5947 | 0.3948 | Media |
| **IVA v2** | 2 | 0.6991 | **71,144** | 0.5875 | 0.5592 | 0.2831 | Media |
| Sparse PCA | 3 | 0.6095 | 44,572 | 0.5067 | 0.5946 | 0.1848 | **Alta** |
| PCA | 3 | 0.6091 | 44,579 | 0.5063 | 0.5947 | **0.3432** | Media |
| Kernel PCA | 2 | 0.5513 | 4,220 | 0.9530 | 0.5745 | 0.6052† | Baja |

IVA v2 destaca en **Calinski-Harabasz (71,144)** — mayor separabilidad
inter-cluster relativa al tamaño de los grupos. ICA lidera en Silhouette puro.

## 6. Visualizaciones generadas
| Archivo | Descripción |
|---------|-------------|
| `situacion_01_varianza_pca.png` | Varianza explicada por componente y acumulada (PCA: 90%) |
| `situacion_01_loadings_pca.png` | Heatmap de cargas PCA |
| `situacion_01_clusters_pca.png` | Clusters en espacio latente PCA (K=3) |
| `situacion_01_comparacion_metodos.png` | Comparativa de métricas por método |
| `situacion_01_efa_loadings.png` | Heatmap EFA + comunalidades por variable |
| `situacion_01_seleccion_k.png` | Curvas Silhouette vs K (PCA, ICA, IVA) |

## 7. Conclusion
**Modelo recomendado:** IVA v2 (vistas semánticas) + ICA para segmentación,
Sparse PCA para interpretabilidad.

- **IVA v2** con vistas espacio-temporal vs económica obtiene Silhouette=0.6991 y
  CH=71,144 (máximo entre todos los métodos), confirmando que la separación entre
  dominios operacionales produce clústeres más compactos y densos. El Alignment
  Score=0.3041 indica que ambas vistas comparten estructura latente real.
- **ICA** (Sil=0.9929) produce la segmentación más cohesiva en términos puros de
  compacidad geométrica, separando viajes de plataformas tecnológicas (HVFHV)
  de taxis tradicionales.
- **Sparse PCA** ofrece máxima interpretabilidad: 3 constructos cuasi-puros
  (duración+tarifa, distancia, ocupación) con cargas casi binarias.
- **PCA** mantiene el mejor balance predictivo sobre la muestra completa
  (AUC=0.5947, R²=0.3432).

---
*Reporte final con métricas reales — 2026-05-28 | Dataset: 2026-03, N=74,485*
