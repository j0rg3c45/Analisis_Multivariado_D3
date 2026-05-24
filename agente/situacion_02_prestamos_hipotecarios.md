# Situación 2: Análisis de Préstamos Hipotecarios (Single-Family Fixed-Rate Loan)

**Dataset:** Single-Family Fixed-Rate Loan Performance Data  
**Objetivo:** Identificar estructuras latentes en un portafolio hipotecario usando IVA y clustering.

---

## 1. Objetivo General

Desarrollar un estudio multivariado avanzado para identificar, modelar y caracterizar las estructuras latentes que explican los patrones de comportamiento crediticio observados en el portafolio hipotecario, utilizando dichas estructuras para generar una segmentación de riesgo interpretable.

## 2. Fases del Estudio

### Fase 1: Construcción del Panel Analítico
- Integrar y preparar tablas de originación, desempeño mensual, atributos del prestatario e inmueble.
- Construir un panel analítico longitudinal y coherente.
- Aptitud: estudio conjunto de variables de riesgo, capacidad de pago y desempeño histórico.

### Fase 2: Extracción de Componentes Latentes (IVA)
- Aplicar Independent Vector Analysis (IVA).
- Extraer componentes multivista que capturen información compartida entre grupos de variables:
  - Dominios del préstamo
  - Dominios del prestatario
  - Dominios del comportamiento mensual
- Fuentes latentes deben explicar la heterogeneidad multivariada del portafolio de manera compacta.

### Fase 3: Evaluación e Interpretación de Componentes
- Evaluar y contrastar componentes IVA vs. indicadores tradicionales de riesgo crediticio.
- Enfoque: interpretabilidad, estabilidad y relevancia para caracterizar patrones de:
  - Morosidad
  - Prepago
  - Incumplimiento
  - Amortización

### Fase 4: Segmentación Basada en Características Latentes
- Usar proyecciones (scores) de componentes IVA como base para clustering.
- Técnicas a explorar:
  - K-means
  - Gaussian Mixture Models (GMM)
  - Clustering jerárquico
- Segmentar préstamos en grupos homogéneos.

### Fase 5: Caracterización de Perfiles de Riesgo
- Interpretar y validar conglomerados resultantes.
- Caracterización en términos de:
  - Perfiles de riesgo
  - Rasgos financieros del prestatario
  - Propiedades estructurales del préstamo

## 3. Resultados Esperados

- Discusión crítica sobre utilidad de segmentaciones para:
  - Gestión de riesgo
  - Monitoreo de cartera
  - Políticas de originación
- Demostración de integración sólida de técnicas avanzadas (IVA + Clustering).
- Capacidad de interpretar patrones complejos en un portafolio hipotecario real.

## 4. Entregables

- Código Python/R/Julia con implementación completa.
- Tablas, gráficos e indicadores de resultados.
- Interpretación de resultados según contexto crediticio.
