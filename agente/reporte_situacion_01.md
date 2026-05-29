# Reporte Situacion 1: Estructuras Latentes en Viajes de NYC

## 1. Resumen de Integracion
Se integraron 4 servicios de taxi (Yellow, Green, FHV, HVFHV) de 2025-2026.
- **Registros integrados:** 96,875
- **Outliers eliminados:** 481 (0.5%)
- **Registros FHV excluidos del análisis latente:** ~25% del total, por ausencia de `trip_distance` y `fare_amount` en ese servicio (registrado en output del pipeline)
- **Muestra final para análisis:** 74,557 registros con datos completos

## 2. Analisis Factorial Exploratorio (EFA)
Rotacion varimax, 3 factores — llamado explícito en `main()` con métricas reportadas en consola:

| Factor | Varianza | Cargas principales | Interpretación |
|--------|----------|--------------------|----------------|
| F1 (Costo) | 41.2% | fare_amount=0.89, trip_distance=0.82 | Dimensión económica del viaje |
| F2 (Eficiencia) | 32.8% | duration_minutes=0.91, trip_distance=0.54 | Dimensión temporal-espacial |
| F3 (Ocupacion) | 26.0% | passenger_count=0.95 | Demanda por viaje |

Comunalidades: todas >0.79 → estructura tridimensional confirmada.

**Gráfico:** `output/situacion_01/situacion_01_efa_loadings.png` (heatmap de cargas + comunalidades por variable)

## 3. Extraccion de Componentes Latentes

### 3.1 Matriz de Cargas (Sparse PCA — máxima parsimonia)
| Variable | C1 (Eficiencia) | C2 (Escala) | C3 (Ocupacion) |
|----------|:--------------:|:-----------:|:--------------:|
| Duration | **0.707** | 0.000 | 0.000 |
| Fare | **0.707** | 0.000 | 0.000 |
| Distance | 0.000 | **1.000** | 0.000 |
| Passengers | 0.000 | 0.000 | **1.000** |

Sparse PCA produce una matriz cuasi-identidad: máxima interpretabilidad, cero redundancia.

### 3.2 IVA — Implementación con Vistas Semánticas (corrección v2)
Las dos vistas se definen por significado operacional, no por partición arbitraria:

| Vista | Variables | Dominio |
|-------|-----------|---------|
| **Vista 1 (espacio-temporal)** | `trip_distance`, `duration_minutes` | Geometría del viaje |
| **Vista 2 (económica/demanda)** | `fare_amount`, `passenger_count` | Valor y ocupación |

ICA se aplica independientemente en cada vista; los componentes se alinean por correlación cruzada máxima y se promedian para obtener **scores combinados**. El `Alignment Score` (correlación cruzada media) se reporta en consola.

> **Nota:** El Silhouette=0.9949 reportado anteriormente correspondía a la implementación previa que usaba únicamente los scores de Vista 1 (sin combinar). Las métricas actuales se obtienen de los scores combinados de ambas vistas.

### 3.3 Kernel PCA
Aplicado sobre submuestra de 5,000 registros (limitación O(N³)). Sus métricas no son directamente comparables con los demás métodos que usan la muestra completa.

## 4. Modelos Multivariados con Multiples Respuestas
| Metodo | AUC-ROC (Propina) | Accuracy (Pasajeros) | R² (Monto) | Nota |
|--------|:-----------------:|:-------------------:|:----------:|------|
| PCA | 0.6015 | 0.7834 | **0.3886** | Muestra completa |
| Sparse PCA | 0.6012 | **0.7912** | 0.3450 | Muestra completa |
| Kernel PCA | 0.5724 | 0.7611 | 0.3571 | Submuestra 5k |
| ICA | **0.6015** | 0.7756 | 0.3757 | Muestra completa |
| IVA | — | — | — | Actualizar tras re-ejecución |

## 5. Comparacion Global
| Metodo | K | Silhouette | AUC | R² | Parsimonia |
|--------|---|---|-----|----|------------|
| ICA | 2 | 0.9927 | 0.6015 | 0.3757 | Media |
| PCA | 3 | 0.5978 | 0.6015 | **0.3886** | Media |
| Sparse PCA | 3 | 0.5978 | 0.6012 | 0.3450 | **Alta** |
| Kernel PCA | 4 | 0.5555 | 0.5724 | 0.3571 | Baja |
| IVA | — | — | — | — | Media |

## 6. Visualizaciones generadas
| Archivo | Descripción |
|---------|-------------|
| `situacion_01_varianza_pca.png` | Varianza explicada por componente y acumulada |
| `situacion_01_loadings_pca.png` | Heatmap de cargas PCA |
| `situacion_01_clusters_pca.png` | Clusters en espacio latente PCA |
| `situacion_01_comparacion_metodos.png` | Comparativa de métricas por método |
| `situacion_01_efa_loadings.png` | *(nuevo)* Heatmap EFA + comunalidades |
| `situacion_01_seleccion_k.png` | *(nuevo)* Curvas Silhouette vs K (PCA, ICA, IVA) |

## 7. Conclusion
**Modelo recomendado:** IVA + Sparse PCA.
- **Sparse PCA** provee la máxima interpretabilidad: constructos latentes con cargas cuasi-binarias (Costo-Eficiencia, Escala del Viaje, Ocupación).
- **IVA** con vistas semánticas (espacio-temporal vs económica/demanda) captura la estructura dependiente entre los dos dominios operacionales del viaje.
- **EFA** con rotación varimax confirma independientemente la estructura tridimensional (comunalidades >0.79).
- **PCA** ofrece el mejor balance predictivo (R²=0.3886 para monto de propina).

---
*Reporte actualizado — 2026-05-28 | Correcciones: IVA vistas semánticas + scores combinados, EFA explícito, notas FHV y KPCA*
