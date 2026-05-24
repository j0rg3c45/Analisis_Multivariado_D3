# Project Instructions - 03-desafio-03

## Organization Rules
- Each situation must have its own Python script inside the `scripts_desafio/` folder (e.g., `scripts_desafio/situacion_01.py`).
- All graphical outputs must be saved in an `output/` directory.
- Within `output/`, there must be a subfolder for each situation (e.g., `output/situacion_01/`).
- After each script execution, a report must be created (e.g., `agente/reporte_situacion_01.md`).
- Questions from the challenge for each situation must be answered explicitly in the report and the final response.
- **Reporting Convention:** All findings, reports, and responses must be sustained with precise numerical figures (e.g., record counts, variance ratios, metric scores) extracted from the analysis.

## Environment
- Use `uv` for all Python tasks (`uv run`, `uv pip`).
- Virtual environment is located in `.venv`.

## Data Paths
- Situación 1: `data/data_situacion_01/`
- Situación 4: `data/data_situacion_04/`

## Situación 4 - Data Context
- **Dataset:** `Landsat_Cubo_Espaciotemporal_UAO.csv`
- **Píxeles:** 60,214
- **Periodo:** 2015-2025 (11 años)
- **Índices:** NDVI, NBR, EVI, NDMI (4 índices × 11 años = 44 variables)
- **Corrección:** Escala Landsat Colección 2 aplicada (factor 0.0000275 - 0.2)
- **Script:** `scripts_desafio/situacion_04.py`
- **Output:** `output/situacion_04/`
- **IVA:** Múltiples pares de vistas (NDVI↔NBR, EVI↔NDMI, NDVI↔EVI)
- **Clustering:** K-means + GMM con comparación de métricas
