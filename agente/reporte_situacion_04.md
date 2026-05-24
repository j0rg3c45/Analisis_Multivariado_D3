# Reporte Situacion 4: Analisis Multitemporal de Cobertura Forestal (Landsat)

## Punto 1: Cubo Espacio-Temporal
- **60,214 pixeles**, periodo 2015-2025 (11 anos)
- **4 indices**: NDVI, NBR, EVI, NDMI (44 variables)
- Correccion de escala Landsat Coleccion 2 aplicada

## Punto 2: Componentes Latentes (PCA/ICA)

### PCA - Varianza Explicada
| PC | Var. Indiv. | Var. Acum. |
|----|:---------:|:---------:|
| PC1 | 75.88% | 75.88% |
| PC2 | 2.87% | 78.74% |
| PC3 | 2.27% | 81.01% |
| PC4 | 2.26% | 83.28% |
| PC5 | 1.95% | **85.22%** |

### ICA - Curtosis
IC1=31,033 | IC2=44,110 | IC3=13,454 | IC4=15.52 | IC5=0.26
-> Fuentes independientes no-gaussianas separadas exitosamente.

## Punto 3: IVA Multivista
- Mejor par: **EVI <-> NDMI** (Alignment Score=0.4828)
- IC5 consistente en todos los pares (corr>0.88)

## Punto 4: Clustering
- K-means seleccionado sobre GMM (Sil=0.2550 vs 0.2242)
- K=4 clusters interpretables ecologicamente

## Punto 5: Evaluacion Cuantitativa
| Cluster | % Area | NDVI | NBR | EVI | NDMI | Tendencia |
|---------|:-----:|:---:|:---:|:---:|:----:|-----------|
| C0 | 15.9% | -2.78% | -3.94% | -3.05% | -4.45% | DEGRADACION |
| C1 | 25.2% | +1.16% | +1.56% | +5.89% | +5.47% | ESTABLE |
| C2 | 23.8% | -0.84% | +8.81% | +5.42% | +147% | REGENERACION |
| C3 | 35.1% | +0.32% | +1.19% | +3.24% | +7.01% | BOSQUE ESTABLE |

## Punto 6: Interpretacion Ecologica
- **Degradacion (15.9%):** Perdida estructural multi-indice, sugiriente de presion antropica
- **Regeneracion (23.8%):** Sucesion vegetal secundaria con fuerte rebrote (NBR +8.81%)
- **Bosque estable (60.3%):** Mayoritaria del area conservada con tendencia positiva

**Limitaciones:** Escala Coleccion 2 aplicada; nubosidad imputada; 1 obs/anual

---
*Reporte final - 2026-05-23*
