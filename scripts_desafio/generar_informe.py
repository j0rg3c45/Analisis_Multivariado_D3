#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genera el informe Markdown para Desafío 3 - Análisis Multivariado.
Salida: Desafio3_Analisis_Multivariado.md  (en la raíz del proyecto)

Para exportar a PDF desde VS Code:
  1. Instalar extensión "Markdown PDF" (yzane.markdown-pdf)
  2. Abrir el .md generado
  3. Ctrl+Shift+P → "Markdown PDF: Export (pdf)"
"""
import os, textwrap

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SCRP = os.path.join(BASE, 'scripts_desafio')
OUT  = os.path.join(BASE, 'Desafio3_Analisis_Multivariado.md')

def img(filename, subfolder, caption=''):
    path = f'output/{subfolder}/{filename}'
    full = os.path.join(BASE, 'output', subfolder, filename)
    if not os.path.exists(full):
        return f'> ⚠ *Figura no disponible: {filename}*\n'
    cap = f'*{caption}*' if caption else ''
    return f'![{caption}]({path})\n\n{cap}\n\n'

def read_script(filename):
    path = os.path.join(SCRP, filename)
    if not os.path.exists(path):
        return f'> ⚠ *Script no encontrado: {filename}*\n'
    with open(path, encoding='utf-8', errors='replace') as f:
        src = f.read()
    lines = src.count('\n') + 1
    return f'**Archivo:** `scripts_desafio/{filename}` ({lines} líneas)\n\n```python\n{src}\n```\n'

# ─────────────────────────────────────────────────────────────────────────────

def portada():
    return """\
# Desafío 3: Identificación de Estructuras Latentes con Métodos Multivariados

---

**Universidad Autónoma de Occidente**
Maestría en Analítica de Big Data — Análisis Multivariado Avanzado

**Autores:**

- Jorge Castaño López
- Natalia Arias Londoño
- Jorge Mario Gallego Uribe
- Jhonathan Leandro Clavijo Troches

**Cali, Colombia — Mayo de 2026**

**Repositorio:** <https://github.com/j0rg3c45/Analisis_Multivariado_D3>

---

<div style="page-break-after: always;"></div>

## Tabla de Contenido

1. [Situación 1: Estructuras Latentes en Viajes de NYC](#situación-1-estructuras-latentes-en-viajes-de-nyc)
   1. [Integración de Datos y Variables de Interés](#11-integración-de-datos-y-variables-de-interés)
   2. [Análisis Factorial Exploratorio (EFA)](#12-análisis-factorial-exploratorio-efa)
   3. [Extracción de Componentes Latentes](#13-extracción-de-componentes-latentes)
   4. [Modelos Multivariados con Múltiples Respuestas](#14-modelos-multivariados-con-múltiples-respuestas)
   5. [Evaluación Comparativa y Clustering](#15-evaluación-comparativa-y-clustering)
   6. [Modelo Final Recomendado](#16-modelo-final-recomendado)
3. [Situación 3: Aislamiento de Firma de Estrés en Voz (SUSAS)](#situación-3-aislamiento-de-firma-de-estrés-en-voz-susas)
   1. [Dataset y Síntesis de Señales](#31-dataset-y-síntesis-de-señales)
   2. [Fase 1: Separación ICA — Escenario Sala de Interrogatorios](#32-fase-1-separación-ica--escenario-sala-de-interrogatorios)
   3. [Fase 2: IVA Multi-Vista — Aislamiento de Firma de Estrés](#33-fase-2-iva-multi-vista--aislamiento-de-firma-de-estrés)
   4. [Comparativa ICA vs IVA e Interpretación](#34-comparativa-ica-vs-iva-e-interpretación)
4. [Situación 4: Análisis Multitemporal de Cobertura Forestal (Landsat)](#situación-4-análisis-multitemporal-de-cobertura-forestal-landsat)
   1. [Construcción del Cubo Multivariado Espacio-Temporal](#41-construcción-del-cubo-multivariado-espacio-temporal)
   2. [Identificación de Componentes Latentes (PCA / ICA)](#42-identificación-de-componentes-latentes-pca--ica)
   3. [Aplicación Obligatoria de IVA](#43-aplicación-obligatoria-de-iva)
   4. [Detección de Zonas de Cambio (Clustering)](#44-detección-de-zonas-de-cambio-clustering)
   5. [Evaluación Cuantitativa del Cambio](#45-evaluación-cuantitativa-del-cambio)
   6. [Interpretación Ecológica y Conclusiones](#46-interpretación-ecológica-y-conclusiones)
- [Apéndice A: Código Python — Situación 1](#apéndice-a-código-python--situación-1)
- [Apéndice B: Código Python — Situación 4](#apéndice-b-código-python--situación-4)
- [Apéndice C: Código Python — Situación 3](#apéndice-c-código-python--situación-3)

<div style="page-break-after: always;"></div>
"""

def s1():
    v  = img('situacion_01_varianza_pca.png',    'situacion_01', 'Varianza explicada por componente y acumulada — PCA (3 componentes capturan el 90%)')
    l  = img('situacion_01_loadings_pca.png',    'situacion_01', 'Heatmap de cargas PCA: relación de las cuatro variables con cada componente principal')
    c  = img('situacion_01_clusters_pca.png',    'situacion_01', 'Clústeres en el espacio latente PCA (K-means, K=3)')
    e  = img('situacion_01_efa_loadings.png',    'situacion_01', 'EFA — Cargas factoriales y comunalidades (varimax, 3 factores)')
    m  = img('situacion_01_comparacion_metodos.png', 'situacion_01', 'Comparativa de métricas predictivas y de clustering por método')
    k  = img('situacion_01_seleccion_k.png',     'situacion_01', 'Curvas Silhouette vs K para PCA, ICA e IVA v2')
    return f"""\
## Situación 1: Estructuras Latentes en Viajes de NYC

A partir de los registros históricos de viajes de la ciudad de Nueva York (datos TLC — marzo 2026),
correspondientes a los servicios *Yellow Taxi*, *Green Taxi*, *For-Hire Vehicles (FHV)* y
*High-Volume FHV (HVFHV)*, se integran y analizan conjuntamente los cuatro conjuntos de datos con
el propósito de identificar y estimar estructuras latentes que expliquen los patrones multivariados
asociados a tres variables de interés operacional: (1) la probabilidad de que un viaje incluya
propina, (2) la cantidad de pasajeros transportados y (3) el monto de propina efectivamente
recibido.

---

### 1.1 Integración de Datos y Variables de Interés

Los cuatro archivos PARQUET fueron cargados y estandarizados en un único dataset unificado.
Los servicios FHV se excluyeron del análisis latente por no registrar `trip_distance` ni
`fare_amount`, conservándose únicamente como parte del conjunto integrado original.

| Etapa | Detalle | Registros |
|-------|---------|----------:|
| Carga inicial (4 servicios) | 25,000 por servicio — muestra balanceada | 100,000 |
| Outliers eliminados | Valores extremos en fare, distance, duration | 550 (0.5%) |
| FHV excluidos del análisis latente | Sin trip_distance ni fare_amount | 24,965 |
| **Muestra final para análisis** | HVFHV=25,000 \\| Green=24,876 \\| Yellow=24,609 | **74,485** |

**Variables de análisis:**

| Variable | Descripción |
|----------|-------------|
| `trip_distance` | Distancia recorrida (km) |
| `duration_minutes` | Duración del viaje en minutos |
| `fare_amount` | Tarifa base del viaje (USD) |
| `passenger_count` | Número de pasajeros |

**Variables de respuesta:** `has_tip` (binaria — AUC-ROC), `passenger_count` (clasificación — Accuracy), `tip_amount` (regresión — R²).

---

### 1.2 Análisis Factorial Exploratorio (EFA)

EFA se aplicó con rotación varimax extrayendo 3 factores, con el objetivo de identificar la
estructura de covarianza subyacente antes de aplicar los métodos de reducción de dimensionalidad.

**Varianza explicada por factor:**

| Factor | Varianza (prop. scores) | Interpretación |
|--------|:-----------------------:|----------------|
| F1 | **95.38%** | Dimensión duración–tarifa (dominante) |
| F2 | 3.77% | Variación residual secundaria |
| F3 | 0.85% | Ruido factorial |

**Comunalidades individuales:**

| Variable | h² | Interpretación |
|----------|----|----------------|
| `duration_minutes` | **0.6055** | Bien representada — carga alta en F1 |
| `fare_amount` | **0.6040** | Bien representada — carga alta en F1 |
| `trip_distance` | 0.0058 | Baja carga: varianza absorbida por `fare_amount` (colinealidad) |
| `passenger_count` | 0.0241 | Dimensión autónoma — prácticamente ortogonal al espacio factorial |

> **Interpretación:** `duration_minutes` y `fare_amount` comparten una estructura factorial
> dominante (h²≈0.60), confirmando la existencia de una dimensión latente de
> *"costo-duración del viaje"*. `trip_distance` tiene h²≈0 porque su varianza queda absorbida
> en la misma dirección que `fare_amount`. `passenger_count` actúa como dimensión ortogonal al
> espacio EFA, coherente con la carga pura = 1.0 del componente C3 en Sparse PCA.

{e}

---

### 1.3 Extracción de Componentes Latentes

#### PCA — Análisis de Componentes Principales

PCA captura la variabilidad total del dataset. Con 3 componentes se explica el **90%** de la
varianza total, identificando las direcciones de máxima dispersión en el espacio de 4 variables.

{v}

{l}

#### Sparse PCA — Constructos Cuasi-Puros (Máxima Parsimonia)

Sparse PCA impone penalización L1 sobre las cargas, produciendo componentes con cargas casi
binarias (0 o ≠ 0). El resultado es la mayor parsimonia interpretativa entre todos los métodos:

| Componente | Variables con carga ≠ 0 | Constructo interpretado |
|------------|------------------------|------------------------|
| C1 | `duration_minutes` + `fare_amount` | *"Costo-Duración del Viaje"* |
| C2 | `trip_distance` | *"Eficiencia Espacial / Distancia"* |
| C3 | `passenger_count` (carga = 1.0) | *"Demanda del Cliente"* — componente puro |

Los constructos de Sparse PCA corresponden directamente a los ejemplos de parsimonia citados
en el desafío: *"Eficiencia del Viaje", "Costo-Distancia", "Demanda del Cliente"*.

#### Kernel PCA

Kernel PCA (RBF) captura relaciones no lineales. Aplicado sobre submuestra de 5,000 registros
por restricciones de memoria (O(N²) en matrices de kernel). R²=0.6052 sobre propina no es
comparable con los demás métodos (submuestra independiente).

#### ICA — Análisis de Componentes Independientes

ICA maximiza la no-gaussianidad de las fuentes latentes, separando contribuciones estadísticamente
independientes. La segmentación ICA (Silhouette=0.9929) separa con máxima cohesión las
plataformas tecnológicas (HVFHV) de los taxis tradicionales (Yellow y Green) en K=2 grupos.

#### IVA v2 — Independent Vector Analysis con Vistas Semánticas

IVA extiende ICA al caso multivista: las 4 variables operacionales se dividen en dos dominios
semánticos que se procesan como vistas independientes, capturando la dependencia entre dominios.

| Vista | Variables | Dominio |
|-------|-----------|---------|
| Vista 1 (espacio-temporal) | `trip_distance`, `duration_minutes` | Geometría del viaje |
| Vista 2 (económica/demanda) | `fare_amount`, `passenger_count` | Valor económico y ocupación |

> **Alignment Score IVA v2: 0.3041** — Las dos vistas comparten ~30% de estructura latente
> (correlación cruzada promedio entre componentes alineados). El vector de alineación [1,1]
> confirma dependencia real entre los dominios del viaje: la duración/distancia correlaciona
> con la tarifa/demanda en el espacio latente.

---

### 1.4 Modelos Multivariados con Múltiples Respuestas

Las proyecciones de cada espacio latente se usaron como predictores en modelos supervisados
para evaluar simultáneamente las 3 variables de respuesta (Logistic Regression para AUC-ROC,
Random Forest para Accuracy y R²):

| Método | Muestra | AUC-ROC (Propina) | Accuracy (Pasajeros) | R² (Monto Propina) |
|--------|:-------:|:-----------------:|:--------------------:|:------------------:|
| PCA | 74,485 | **0.5947** | 1.0000 (*) | 0.3432 |
| Sparse PCA | 74,485 | **0.5946** | 0.9999 (*) | 0.1848 |
| Kernel PCA | 5,000 (†) | 0.5745 | 0.9920 | **0.6052** (†) |
| ICA | 74,485 | **0.5947** | 0.9999 (*) | **0.3948** |
| **IVA v2** | 74,485 | 0.5592 | **0.9028** | 0.2831 |

> (*) Accuracy ≈ 1.0: el dataset tiene dominio absoluto de viajes de 1 pasajero (~98%+).
> Un clasificador trivial ("siempre 1") alcanza Acc ≈ 98%. La Acc = **0.9028** de IVA v2 es la
> más informativa porque el espacio IVA tiene menor concentración trivial al separar
> `fare_amount` y `passenger_count` en vistas distintas.
>
> (†) Kernel PCA sobre submuestra de 5,000 registros — R² no comparable con los demás.

{m}

---

### 1.5 Evaluación Comparativa y Clustering

Sobre cada espacio latente se aplica K-means optimizando K mediante Silhouette Score,
Calinski-Harabasz (CH) y Davies-Bouldin (DB):

| Método | K | Silhouette | Calinski-Harabasz | Davies-Bouldin | Parsimonia |
|--------|:-:|:----------:|:-----------------:|:--------------:|:----------:|
| ICA | 2 | **0.9929** | 34,896 | **0.1324** | Media |
| **IVA v2** | 2 | 0.6991 | **71,144** | 0.5875 | Media |
| Sparse PCA | 3 | 0.6095 | 44,572 | 0.5067 | **Alta** |
| PCA | 3 | 0.6091 | 44,579 | 0.5063 | Media |
| Kernel PCA | 2 | 0.5513 | 4,220 | 0.9530 | Baja |

{c}

{k}

---

### 1.6 Modelo Final Recomendado

> **Modelo recomendado: IVA v2 (vistas semánticas) + ICA para segmentación + Sparse PCA para interpretabilidad**

La respuesta a la pregunta central — *¿cuál es el modelo de estructura factorial más robusto,
interpretable y válido para caracterizar la dinámica subyacente en los datos de viajes?* — se
evalúa en los tres ejes solicitados:

**1. Interpretabilidad y Parsimonia:**
Sparse PCA produce los constructos más parsimoniosos: *C1="Costo-Duración"*, *C2="Distancia"*,
*C3="Demanda"* (carga pura = 1.0). EFA confirma que `duration_minutes` y `fare_amount` comparten
estructura factorial (h²≈0.60), validando la agrupación de C1. Los constructos son cuasi-puros,
directamente interpretables y alineados con los ejemplos del desafío.

**2. Coherencia de Segmentación:**
IVA v2 obtiene el Calinski-Harabasz máximo (**71,144**) — mayor varianza inter-cluster relativa
a la intra-cluster entre todos los métodos. El Alignment Score = 0.3041 valida la dependencia
real entre el dominio espacio-temporal y el económico-operacional. ICA lidera en Silhouette puro
(**0.9929**) y Davies-Bouldin mínimo (**0.1324**), separando plataformas tecnológicas de taxis
tradicionales con la mayor cohesión geométrica.

**3. Validez Predictiva:**
PCA e ICA alcanzan el mayor AUC-ROC (**0.5947**) para predecir la probabilidad de propina.
ICA obtiene el mayor R² (**0.3948**) para el monto de propina. IVA v2 produce la Accuracy más
informativa en clasificación de pasajeros (**0.9028** vs. ≈ 1.0 triviales en los demás métodos).
PCA captura el 90% de la varianza original con solo 3 componentes.

<div style="page-break-after: always;"></div>
"""

def s2():
    return ''

def s3():
    f1w = img('situacion_03_fase1_formas_onda.png',    'situacion_03', 'Fase 1 — Formas de onda: fuentes S₁/S₂, mezclas M₁/M₂ y componentes recuperados C₁/C₂')
    f1s = img('situacion_03_fase1_psd_sir.png',        'situacion_03', 'Fase 1 — Análisis espectral (PSD 0–600 Hz) y Signal-to-Interference Ratio (SIR)')
    f2c = img('situacion_03_fase2_componentes_iva.png','situacion_03', 'Fase 2 — Vistas multi-condición D₁/D₂/D₃ y componentes IVA recuperados IC1/IC2/IC3')
    f2h = img('situacion_03_fase2_heatmap_zcr.png',    'situacion_03', 'Fase 2 — Mapa de correlaciones IVA y perfil ZCR por condición')
    return f"""\
## Situación 3: Aislamiento de Firma de Estrés en Voz (SUSAS)

Se desarrolla un análisis de separación de fuentes de voz bajo diferentes condiciones de estrés
fisiológico, replicando las condiciones del corpus **SUSAS (Speech Under Simulated and Actual
Stress, LDC99S78)**. El análisis opera en dos fases complementarias: ICA para separación de
fuentes en un escenario de mezcla lineal (Fase 1), e IVA multi-vista para aislar la *firma
de estrés* de la *identidad vocal* a través de tres condiciones simultáneas (Fase 2).

---

### 3.1 Dataset y Síntesis de Señales

El corpus SUSAS registra voz de 32 hablantes bajo condiciones documentadas de estrés acústico
y cognitivo. Dado que el acceso al corpus requiere licencia LDC, las señales se sintetizan con
los parámetros fisiológicos publicados de SUSAS: suma de armónicos (F₀, 2F₀, 3F₀, 4F₀),
modulación de envolvente y tremor laríngeo.

**Parámetros de síntesis:**

| Condición SUSAS | F₀ | Tremor (Hz / Amp) | Jitter | Ruido | Envolvente |
|----------------|:--:|:-----------------:|:------:|:-----:|:----------:|
| Neutral | 120 Hz | 0.5 Hz / Amp=2 | Nulo | 1% | mod=0.25, f=3.0 Hz |
| cond50 (50% carga cognitiva) | 138 Hz (+15%) | 3.0 Hz / Amp=6 | Bajo | 2.5% | mod=0.33, f=4.0 Hz |
| cond70 / Lombard (85 dB SPL) | 162 Hz (+35%) | 5.0 Hz / Amp=10 | Alto (200 pts) | 5% | mod=0.42, f=5.5 Hz |

**Configuración de simulación:** SR = 8,000 Hz (igual que SUSAS) · Duración = 2.0 s ·
16,000 muestras · Semilla aleatoria = 42

| Señal | F₀ | RMS |
|-------|----|-----|
| S₁ (neutro) | 120 Hz | 0.4344 |
| S₂ (estrés cond70) | 162 Hz | 0.3141 |
| D₂ (cond50) | 138 Hz | 0.3986 |

---

### 3.2 Fase 1: Separación ICA — Escenario Sala de Interrogatorios

**Escenario:** Dos micrófonos capturan mezclas lineales de S₁ (voz neutra) y S₂ (voz
estresada cond70/Lombard):

| Micrófono | Mezcla |
|-----------|--------|
| M₁ | 0.7 · S₁ + 0.3 · S₂ |
| M₂ | 0.4 · S₁ + 0.6 · S₂ |

**Método:** FastICA (algoritmo deflation, whiten='unit-variance', max_iter=1000, tol=1e-5,
seed=42).

**Tabla 1 — Correlaciones |r| entre componentes ICA y fuentes originales:**

| | S₁ neutro | S₂ estrés |
|---|:---------:|:---------:|
| C₁ | **1.0000** | 0.0026 |
| C₂ | 0.0032 | **1.0000** |

**Métricas de separación:**

| Componente | SIR (dB) | Interpretación |
|------------|:--------:|----------------|
| C₁ (neutro) | **49.83 dB** | Separación perfecta — supera umbral práctico (20 dB) |
| C₂ (estrés) | **51.62 dB** | Separación perfecta — resolución ≈ 50 dB |

> **Interpretación:** FastICA recupera perfectamente ambas fuentes (|r|=1.0000, SIR≈50 dB).
> El algoritmo aprovecha la **no-gaussianidad diferencial** entre las señales: S₁ (voz neutra,
> kurtosis más baja) y S₂ (voz estresada con tremor intenso y jitter, kurtosis alta).
> La gran diferencia en F₀ (120 vs 162 Hz) y la presencia de perturbaciones discretas (jitter)
> en S₂ maximizan la separabilidad estadística de las fuentes.

{f1w}

*Fase 1 — Formas de onda: fuentes originales, mezclas de micrófono y componentes ICA recuperados*

{f1s}

*Fase 1 — PSD comparativa (armónicos F₀ neutro 120 Hz vs estrés 162 Hz) y SIR por componente*

---

### 3.3 Fase 2: IVA Multi-Vista — Aislamiento de Firma de Estrés

**Escenario:** Mismo hablante, misma palabra, tres condiciones de estrés como vistas
independientes. El objetivo es separar la **identidad vocal** (invariante al estrés) de la
**modulación fisiológica** (dependiente del nivel de estrés).

| Vista | Condición SUSAS | F₀ | Tremor |
|-------|----------------|:--:|:------:|
| D₁ | Neutral | 120 Hz | mínimo |
| D₂ | cond50 (50% carga cognitiva) | 138 Hz | moderado |
| D₃ | cond70 / Lombard (85 dB SPL) | 162 Hz | alto + jitter |

**Método IVA aproximado:** blanqueo z-score individual por vista + FastICA sobre la matriz
aumentada [D₁|D₂|D₃] (n_components=3, deflation, max_iter=2000, tol=1e-6, seed=42).

**Tabla 2 — Correlaciones |r| entre componentes IVA y vistas originales:**

| Componente | r(D₁ neutro) | r(D₂ bajo) | r(D₃ alto) | Etiqueta IVA |
|------------|:------------:|:----------:|:----------:|:-------------|
| IC1 | 0.0106 | 0.0204 | **0.9994** | *Firma de estrés (alta carga)* |
| IC2 | **0.9999** | 0.0055 | 0.0098 | *Identidad vocal (neutro)* |
| IC3 | 0.0057 | **0.9998** | 0.0318 | *Modulación de estrés (baja carga)* |

**Tabla 3 — ICA individual: correlaciones cross-vista (línea base comparativa):**

| Par | Correlación |r| | Interpretación |
|-----|:-----------:|----------------|
| IC_D₁ ↔ IC_D₂ | 0.0003 | Sin estructura compartida |
| IC_D₁ ↔ IC_D₃ | 0.0006 | Sin estructura compartida |
| IC_D₂ ↔ IC_D₃ | 0.0114 | Sin estructura compartida |

> **ICA individual NO puede aislar la firma de estrés cross-vista:** cada señal se procesa de
> forma independiente y el resultado es ortogonal a las demás condiciones (r ≈ 0). IVA, al
> procesar las tres vistas simultáneamente, identifica la dependencia estadística entre condiciones
> del mismo hablante y separa con |r|≥0.9994 cada componente hacia su condición específica.

**Perfil acústico — Tasa de Cruce por Cero (ZCR, proxy de F₀):**

| Condición | ZCR media | ZCR ±1σ | Variación respecto a D₁ |
|-----------|:---------:|:-------:|:----------------------:|
| D₁ neutro | 239.1 Hz | ±14.9 Hz | — |
| D₂ estrés bajo | 284.3 Hz | ±26.6 Hz | +18.9% |
| D₃ estrés alto | 409.0 Hz | ±72.3 Hz | +71.1% |

> El ZCR crece con el nivel de estrés de forma no lineal: el salto de D₂→D₃ (+44.6%) supera
> al de D₁→D₂ (+18.9%), reflejando la combinación de mayor F₀, tremor intenso y jitter
> supraglótico en la condición Lombard (85 dB SPL). La variabilidad (σ) también se triplica
> de D₁ a D₃, indicador de inestabilidad fonatoria bajo estrés extremo.

{f2c}

*Fase 2 — Señales de las tres condiciones SUSAS y componentes IVA: firma de estrés, identidad vocal y modulación*

{f2h}

*Fase 2 — Heatmap |r| componentes IVA × vistas y ZCR por trama mostrando crecimiento de F₀ con estrés*

---

### 3.4 Comparativa ICA vs IVA e Interpretación

**Resumen de desempeño:**

| Método | Tarea | Métrica clave | Resultado |
|--------|-------|:-------------:|-----------|
| **ICA** | Separación de 2 fuentes mezcladas | SIR | **49.8–51.6 dB** (perfecto) |
| **ICA individual** | Caracterización cross-vista | r cross-vista | 0.0003–0.0114 (nulo) |
| **IVA** | Aislamiento de firma de estrés | r por componente | **0.9994–0.9999** (perfecto) |

> **Conclusión:** ICA es óptimo cuando las fuentes se mezclan físicamente (reverberación,
> posición de micrófonos) y las señales tienen estadísticas muy diferentes. IVA es necesario
> cuando el objetivo es *aislar un factor latente* — en este caso, la firma de estrés fisiológico
> — que **no aparece en ninguna vista individual** sino únicamente en la **dependencia estadística
> entre vistas**. La identidad vocal (IC2, |r|=0.9999 con D₁) es el componente estacionario
> invariante al estrés; la firma de estrés alto (IC1, |r|=0.9994 con D₃) captura la activación
> del sistema nervioso autónomo: aumento de F₀ (+35%), tremor laríngeo (5 Hz), jitter
> supraglótico y mayor energía de alta frecuencia documentados en los 32 hablantes de SUSAS.

**Implicaciones para diagnóstico clínico:**

- El componente IVA de identidad vocal (IC2) puede usarse como **huella biométrica** robusta al
  estrés — invariante a la condición emocional del hablante.
- La firma de estrés alto (IC1) es el indicador más sensible para detectar estados de Lombard
  (entornos ruidosos), con ZCR +71% respecto al baseline.
- La modulación de estrés bajo (IC3) es diagnóstico de carga cognitiva sostenida (cond50),
  relevante para evaluación de fatiga en pilotos, controladores de tráfico aéreo y operadores.

<div style="page-break-after: always;"></div>
"""

def s4():
    v  = img('situacion_04_varianza_pca.png',          'situacion_04', 'Varianza explicada por componente PCA (barras) y acumulada (línea roja)')
    ml = img('situacion_04_mapas_latentes.png',        'situacion_04', 'Mapas espaciales de PC1 (vigor base), PC2 (variabilidad estacional) e IC1 (perturbaciones abruptas)')
    cc = img('situacion_04_comparacion_clustering.png','situacion_04', 'Silhouette Score vs K — K-means vs GMM sobre el espacio IVA (EVI ↔ NDMI)')
    mc = img('situacion_04_mapa_clusters.png',         'situacion_04', 'Mapa temático de clústeres: distribución espacial de trayectorias de cambio forestal (K=4, K-means)')
    ti = img('situacion_04_trayectorias_indices.png',  'situacion_04', 'Trayectorias temporales (2015–2025) de los 4 índices por clúster')
    tn = img('situacion_04_trayectorias_ndvi.png',     'situacion_04', 'Trayectorias NDVI con bandas ±1σ por clúster y tasa anual de cambio en leyenda')
    ci = img('situacion_04_cambio_indices.png',        'situacion_04', 'Cambio total (%) y tasa anual (%) por índice y clúster — comparativa multi-índice')
    return f"""\
## Situación 4: Análisis Multitemporal de Cobertura Forestal (Landsat)

Se desarrolla un estudio multivariado para identificar, aislar y caracterizar las estructuras
latentes que explican los patrones de cambio en la cobertura forestal, utilizando datos
multitemporales del archivo Landsat Science (USGS).

---

### 4.1 Construcción del Cubo Multivariado Espacio-Temporal

#### Dataset y Origen de los Datos

El archivo `Landsat_Cubo_Espaciotemporal_UAO.csv` contiene el cubo de datos espaciotemporal
construido a partir de imágenes **Landsat 5, 7, 8 y 9** para un área de interés en el contexto
UAO. Las bandas espectrales relevantes (**Rojo, Infrarrojo Cercano — NIR, Infrarrojo de Onda
Corta — SWIR**) fueron extraídas en la pipeline de datos USGS, a partir de las cuales se
derivaron los cuatro índices de vegetación requeridos para el análisis:

- **NDVI** (*Normalized Difference Vegetation Index*): salud vegetativa general — (NIR−Rojo)/(NIR+Rojo)
- **EVI** (*Enhanced Vegetation Index*): vigor corregido por atmósfera y suelo
- **NBR** (*Normalized Burn Ratio*): severidad de incendios / biomasa — (NIR−SWIR)/(NIR+SWIR)
- **NDMI** (*Normalized Difference Moisture Index*): contenido de humedad — (NIR−SWIR)/(NIR+SWIR)

#### Características del Cubo

| Parámetro | Valor |
|-----------|-------|
| Total de píxeles (observaciones) | 60,214 |
| Período temporal | 2015 – 2025 (11 años) |
| Índices de vegetación | NDVI, NBR, EVI, NDMI |
| Variables totales por píxel | 44 (4 índices × 11 años) |
| Valores faltantes (nubosidad) | Imputados con mediana por columna |

#### Validación de Escala en Tiempo de Ejecución

El script detecta automáticamente si los valores son DN (enteros Landsat, máx > 1.5) y aplica
la corrección oficial Landsat Collection 2: `reflectancia = DN × 0.0000275 − 0.2`. Si ya están
en escala normalizada, los usa directamente sin aplicar la corrección (evita doble transformación).

**Resultado en ejecución real (2026-05-28):**

| Índice | Resultado |
|--------|-----------|
| EVI | Corrección Landsat C2 aplicada (valores en formato DN) |
| NDVI | Corrección Landsat C2 aplicada (valores en formato DN) |
| NBR | Valores ya en escala de reflectancia normalizada |
| NDMI | Valores ya en escala de reflectancia normalizada |

El CSV contiene formatos mixtos (EVI y NDVI en DN; NBR y NDMI ya normalizados). El script
manejó esta heterogeneidad automáticamente sin intervención manual.

---

### 4.2 Identificación de Componentes Latentes (PCA / ICA)

#### PCA — Modos de Variabilidad Temporal

| Componente | Var. Individual | Var. Acumulada | Interpretación ecológica |
|------------|:--------------:|:--------------:|--------------------------|
| PC1 | **75.88%** | 75.88% | Vigor forestal base (estado vegetativo dominante) |
| PC2 | 2.87% | 78.74% | Variabilidad estacional / interanual |
| PC3 | 2.27% | 81.01% | Cambios interanuales progresivos |
| PC4 | 2.26% | 83.28% | Nubosidad / ruido residual |
| PC5 | 1.95% | **85.22%** | Perturbaciones locales puntuales |

PC1 (75.88%) domina la varianza total: captura el estado vegetativo base de cada píxel a lo
largo de toda la serie temporal. La caída abrupta de PC2 en adelante (<3%) indica que la mayor
parte de la información en el cubo es el *estado* de la vegetación, no su *cambio* temporal.

#### ICA — Separación de Fuentes Independientes de Disturbio

| Componente | Curtosis | Interpretación ecológica |
|------------|:--------:|--------------------------|
| IC1 | 31,033 | Perturbaciones abruptas — incendios, tala intensiva |
| IC2 | **44,110** | Eventos extremos de cambio (máxima no-gaussianidad) |
| IC3 | 13,454 | Degradación progresiva multi-año |
| IC4 | 15.52 | Ruido residual estructurado |
| IC5 | 0.26 | Componente gaussiana — ruido de fondo |

La curtosis extrema (>10,000) en IC1–IC3 confirma la separación exitosa de fuentes
no-gaussianas, típicamente asociadas a eventos discretos de disturbio forestal.
IC5 (curtosis ≈ 0) representa ruido estadístico distribuido normalmente.

{v}

{ml}

---

### 4.3 Aplicación Obligatoria de IVA

IVA se aplica tratando pares de índices de vegetación como "vistas" independientes de las mismas
fuentes latentes de cambio. Se exploraron tres pares de vistas:

| Par de Vistas | Alignment Score | IC más alineado (correlación cruzada) |
|---------------|:--------------:|---------------------------------------|
| NDVI ↔ NBR | 0.2234 | IC5 (corr = 0.8937) |
| **EVI ↔ NDMI** | **0.4828** | IC5 (corr = 0.8820) |
| NDVI ↔ EVI | 0.2122 | IC5 (corr = 0.9283) |

> **Mejor par: EVI ↔ NDMI (Alignment Score = 0.4828).** EVI captura el vigor vegetativo
> corregido; NDMI mide el contenido de humedad. Ambos índices responden a los mismos procesos
> ecológicos (degradación = caída simultánea; regeneración = subida simultánea), lo que
> produce la mayor dependencia multivista. IC5 tiene correlación cruzada > 0.88 en los tres
> pares, representando el factor latente común del estado base de la vegetación.
>
> Los *scores* combinados del par EVI ↔ NDMI se usaron como input para el clustering.

---

### 4.4 Detección de Zonas de Cambio (Clustering)

Se evaluaron K-means y GMM sobre el espacio IVA, optimizando K mediante Silhouette Score
en submuestra de 5,000 píxeles:

| Método | K óptimo (Silhouette) | K=4 Silhouette | Davies-Bouldin | Calinski-Harabasz |
|--------|:---------------------:|:--------------:|:--------------:|:-----------------:|
| **K-means** | K=2 (0.3224) | **0.2550** | **1.4858** | **19,699** |
| GMM | K=3 (0.2477) | 0.2242 | 3.3412 | 10,040 |

K-means supera a GMM en las tres métricas. Se adoptó **K=4** sobre el K óptimo estadístico (K=2)
por razones de interpretabilidad ecológica: K=4 permite distinguir degradación, regeneración
incipiente y bosque maduro como categorías ecológicamente significativas y operativamente útiles.

{cc}

{mc}

---

### 4.5 Evaluación Cuantitativa del Cambio

**Resumen de cambios totales 2015–2025:**

| Clúster | % Área | Píxeles | ΔNDVI | ΔNBR | ΔEVI | ΔNDMI | Clasificación |
|---------|:------:|:-------:|:-----:|:----:|:----:|:-----:|:-------------:|
| C0 | 15.9% | 9,594 | −1.00% | −2.10% | −4.26% | −3.39% | **DEGRADACIÓN** |
| C1 | 25.2% | 15,179 | +4.29% | +6.17% | +5.86% | +11.97% | **REG. ACTIVA** |
| C2 | 23.8% | 14,305 | +5.34% | +26.80% | +9.76% | +240%(*) | **REG. ACTIVA** |
| C3 | 35.1% | 21,136 | +4.16% | +6.59% | +4.99% | +13.67% | **REG. ACTIVA** |

> (*) NDMI C2: cambio porcentual inflado por denominador ≈ 0 (inicio: −0.009, fin: +0.013).
> El cambio absoluto (+0.022 unidades) es el dato ecológicamente relevante.

**Detalle por clúster — valores inicio (2015) → fin (2025) y tasa anual:**

**C0 — DEGRADACIÓN (15.9% — 9,594 píxeles):**

| Índice | 2015 | 2025 | Cambio Total | Tasa Anual |
|--------|-----:|-----:|:------------:|:----------:|
| NDVI | 0.7145 | 0.7073 | −1.00% | −0.10%/año |
| NBR | 0.5693 | 0.5573 | −2.10% | −0.21%/año |
| EVI | 0.4896 | 0.4687 | −4.26% | −0.43%/año |
| NDMI | 0.3119 | 0.3014 | −3.39% | −0.34%/año |

**C1 — BOSQUE DENSO EN REGENERACIÓN (25.2% — 15,179 píxeles):**

| Índice | 2015 | 2025 | Cambio Total | Tasa Anual |
|--------|-----:|-----:|:------------:|:----------:|
| NDVI | 0.6803 | 0.7095 | +4.29% | +0.42%/año |
| NBR | 0.5176 | 0.5495 | +6.17% | +0.60%/año |
| EVI | 0.4459 | 0.4720 | +5.86% | +0.57%/año |
| NDMI | 0.2577 | 0.2886 | +11.97% | +1.14%/año |

**C2 — REGENERACIÓN INCIPIENTE / BAJA COBERTURA (23.8% — 14,305 píxeles):**

| Índice | 2015 | 2025 | Cambio Total | Tasa Anual |
|--------|-----:|-----:|:------------:|:----------:|
| NDVI | 0.3026 | 0.3188 | +5.34% | +0.52%/año |
| NBR | 0.1062 | 0.1346 | +26.80% | +2.40%/año |
| EVI | 0.1888 | 0.2072 | +9.76% | +0.94%/año |
| NDMI | −0.0092 | +0.0129 | +240%(*) | +3.47%/año |

**C3 — BOSQUE MADURO EN RECUPERACIÓN (35.1% — 21,136 píxeles):**

| Índice | 2015 | 2025 | Cambio Total | Tasa Anual |
|--------|-----:|-----:|:------------:|:----------:|
| NDVI | 0.6763 | 0.7045 | +4.16% | +0.41%/año |
| NBR | 0.5004 | 0.5334 | +6.59% | +0.64%/año |
| EVI | 0.4433 | 0.4654 | +4.99% | +0.49%/año |
| NDMI | 0.2427 | 0.2758 | +13.67% | +1.29%/año |

{ti}

{tn}

{ci}

---

### 4.6 Interpretación Ecológica y Conclusiones

> **Síntesis:** El **84.1%** del área (C1+C2+C3) muestra tendencias positivas en todos los
> índices, indicando predominio de procesos de recuperación y estabilidad forestal. El **15.9%**
> restante (C0) presenta degradación estructural sostenida.

**C0 — Degradación (15.9%):** Pérdida consistente en los cuatro índices (EVI −4.26%,
NDMI −3.39%, NBR −2.10%, NDVI −1.00%). La declinación simultánea en vigor, humedad y biomasa
descarta variación fenológica estacional: es indicativa de degradación estructural sostenida.
La concentración geográfica sugiere presión antropogénica focalizada: expansión agrícola,
tala selectiva o urbanización.

**C2 — Regeneración incipiente (23.8%):** Zona de baja cobertura inicial (NDVI ≈ 0.30, menos
de la mitad que el bosque denso). NBR +26.80% (recuperación de biomasa) y NDMI transitando de
negativo a positivo son señales diagnósticas de **sucesión vegetal secundaria** en áreas
previamente deforestadas o quemadas antes de 2015.

**C1 y C3 — Bosque en recuperación activa (60.3%):** Alta cobertura inicial (NDVI ≈ 0.68–0.71)
con mejora sostenida en los cuatro índices. NDMI es el índice más sensible para capturar la
recuperación temprana del dosel (C1: +11.97%, C3: +13.67%) *antes* de que los cambios sean
evidentes en NDVI — lo que tiene implicaciones para el diseño de programas de monitoreo
(priorizar NDMI como indicador temprano).

**Consistencia multi-índice:** La concordancia direccional entre NDVI, NBR, EVI y NDMI dentro
de cada clúster valida la robustez de la segmentación. No existe ningún clúster con tendencias
contradictorias entre índices, descartando artefactos de procesamiento.

**Limitaciones:**
- Escala mixta en el CSV original (EVI/NDVI en DN; NBR/NDMI normalizados) — manejada automáticamente.
- Resolución temporal: 1 observación anual — impide detectar eventos intra-anuales (incendios estacionales).
- NDMI de C2: cambio porcentual inflado por denominador inicial ≈ 0; el cambio absoluto es el dato relevante.

**Aplicaciones:**
- Focalizar inspecciones de campo en C0 (degradación activa — 15.9% del área).
- Evaluar efectividad de programas de reforestación en C2 (regeneración incipiente).
- Usar NDMI como indicador temprano de recuperación del dosel en C1 y C3.
- Los mapas de clústeres permiten priorizar áreas para políticas de conservación diferenciadas por perfil de cambio.

<div style="page-break-after: always;"></div>
"""

def apendice_a():
    return f"""\
## Apéndice A: Código Python — Situación 1

Script de análisis de estructuras latentes en datos de viajes NYC (*Yellow, Green, FHV, HVFHV*).
Incluye carga de datos PARQUET, integración de 4 servicios, PCA, Sparse PCA, Kernel PCA, ICA,
IVA v2 con vistas semánticas, EFA con rotación varimax, clustering K-means y generación de
visualizaciones.

{read_script('situacion_01.py')}

<div style="page-break-after: always;"></div>
"""

def apendice_b():
    return f"""\
## Apéndice B: Código Python — Situación 4

Script de análisis multitemporal de cobertura forestal con datos Landsat (cubo espaciotemporal
2015–2025). Incluye validación de escala en runtime (Landsat Collection 2), PCA, ICA, IVA
multivista (3 pares de índices de vegetación), clustering K-means vs GMM, evaluación cuantitativa
del cambio y generación de mapas temáticos y trayectorias temporales.

{read_script('situacion_04.py')}

<div style="page-break-after: always;"></div>
"""

def apendice_c():
    return f"""\
## Apéndice C: Código Python — Situación 3

Script de separación de fuentes de voz bajo estrés con datos sintéticos basados en SUSAS
(LDC99S78). Incluye síntesis de señales con parámetros fisiológicos documentados, Fase 1
(FastICA sobre mezcla lineal de dos micrófonos), Fase 2 (IVA multi-vista sobre tres
condiciones de estrés), métricas SIR y correlación, ZCR como proxy de F₀ y generación
de visualizaciones.

{read_script('situacion_03.py')}
"""

# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('Generando informe Markdown...')
    secciones = [
        portada(),
        s1(),
        s2(),
        s3(),
        s4(),
        apendice_a(),
        apendice_b(),
        apendice_c(),
    ]
    contenido = '\n'.join(secciones)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(contenido)
    size_kb = os.path.getsize(OUT) / 1024
    print(f'Archivo generado: {OUT}')
    print(f'Tamaño: {size_kb:.0f} KB')
    print()
    print('Para exportar a PDF desde VS Code:')
    print('  1. Instalar extensión "Markdown PDF" (yzane.markdown-pdf) si no la tiene')
    print('  2. Abrir Desafio3_Analisis_Multivariado.md en VS Code')
    print('  3. Ctrl+Shift+P -> escribir "Markdown PDF: Export (pdf)" -> Enter')
    print('     (o clic derecho en el editor -> "Markdown PDF: Export (pdf)")')
    print()
    print('Configuración recomendada en settings.json:')
    print('  "markdown-pdf.format": "A4"')
    print('  "markdown-pdf.margin.top": "1.5cm"')
    print('  "markdown-pdf.margin.bottom": "1.5cm"')
    print('  "markdown-pdf.margin.left": "2cm"')
    print('  "markdown-pdf.margin.right": "2cm"')

if __name__ == '__main__':
    main()
