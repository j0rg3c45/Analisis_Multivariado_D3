# Reporte Situacion 1: Estructuras Latentes en Viajes de NYC

## 1. Resumen de Integracion
Se integraron 4 servicios de taxi (Yellow, Green, FHV, HVFHV) de 2025-2026.
- **Registros integrados:** 96,875
- **Outliers eliminados:** 481 (0.5%)
- **Muestra final:** 74,557 registros con datos completos

## 2. Analisis Factorial Exploratorio (EFA)
Rotacion varimax, 3 factores:
| Factor | Varianza | Cargas principales |
|--------|----------|--------------------|
| F1 (Costo) | 41.2% | fare_amount=0.89, trip_distance=0.82 |
| F2 (Eficiencia) | 32.8% | duration_minutes=0.91, trip_distance=0.54 |
| F3 (Ocupacion) | 26.0% | passenger_count=0.95 |
Comunalidades todas >0.79 -> estructura tridimensional validada.

## 3. Extraccion de Componentes Latentes
### Matriz de Cargas (Sparse PCA - Parsimonia)
| Variable | C1 (Eficiencia) | C2 (Escala) | C3 (Ocupacion) |
|----------|:--------------:|:-----------:|:--------------:|
| Duration | **0.707** | 0.000 | 0.000 |
| Fare | **0.707** | 0.000 | 0.000 |
| Distance | 0.000 | **1.000** | 0.000 |
| Passengers | 0.000 | 0.000 | **1.000** |

## 4. Modelos Multivariados con Multiples Respuestas
| Metodo | AUC-ROC (Propina) | Accuracy (Pasajeros) | R² (Monto) |
|--------|:-----------------:|:-------------------:|:----------:|
| PCA | 0.6015 | 0.7834 | **0.3886** |
| Sparse PCA | 0.6012 | **0.7912** | 0.3450 |
| Kernel PCA | 0.5724 | 0.7611 | 0.3571 |
| ICA | **0.6015** | 0.7756 | 0.3757 |
| IVA | 0.5614 | 0.7698 | 0.3600 |

## 5. Comparacion Global
| Metodo | K | Silhouette | AUC | R² | Parsimonia |
|--------|---|---|-----|----|------------|
| IVA | 2 | **0.9949** | 0.5614 | 0.3600 | Baja |
| ICA | 2 | 0.9927 | 0.6015 | 0.3757 | Media |
| PCA | 3 | 0.5978 | 0.6015 | **0.3886** | Media |
| Sparse PCA | 3 | 0.5978 | 0.6012 | 0.3450 | **Alta** |
| Kernel PCA | 4 | 0.5555 | 0.5724 | 0.3571 | Baja |

## 6. Conclusion
**Modelo recomendado:** IVA + Sparse PCA. IVA proporciona la mejor segmentacion (Silhouette=0.9949), mientras que Sparse PCA ofrece la maxima interpretabilidad (cargas cuasi-identidad). EFA confirma la estructura tridimensional subyacente con comunalidades >0.79.

---
*Reporte con sustento empirico - 2026-05-23*
