# Situación 3: Aislamiento de Firma de Estrés en Señales de Voz (SUSAS)

**Dataset:** SUSAS (Speech Under Simulated and Actual Stress)  
**Objetivo:** Evaluar capacidad de ICA e IVA para aislar y caracterizar la "firma del estrés" como componente independiente de la señal de voz.

---

## Fase 1: Aplicación de ICA

Simular grabación de 2 canales en sala de interrogatorios: "interrogador" (calmado) y "sujeto" (bajo estrés).

### Pasos:
1. **Preparación:** Seleccionar 2 archivos de audio del dataset SUSAS:
   - **S1 (Fuente 1):** Grabación "Neutra" (sin estrés)
   - **S2 (Fuente 2):** Grabación "Estresada" (High Task Load o Lombard Effect)

2. **Simulación de mezcla:** Crear 2 "grabaciones de micrófono" (M1, M2) mediante mezcla lineal:
   - `M1 = 0.7·S1 + 0.3·S2`
   - `M2 = 0.4·S1 + 0.6·S2`

3. **Pregunta:** Aplicando ICA, ¿en qué medida es posible separar y recuperar las señales originales (S1, S2) a partir de las mezclas (M1, M2)?
   - Evaluar calidad de separación:
     - **Cualitativa:** auditiva
     - **Cuantitativa:** métricas como SIR (Signal-to-Interference Ratio)

## Fase 2: Aislamiento de Firma de Estrés (IVA)

Usar estructura multivista de SUSAS. Cada condición de estrés es una "vista" diferente.

### Pasos:
1. **Preparación:** Seleccionar múltiples grabaciones del mismo hablante diciendo la misma palabra bajo diferentes condiciones:
   - **Vista 1 (D1):** Grabación "Neutra"
   - **Vista 2 (D2):** Grabación "Estresada - Tarea Baja"
   - **Vista 3 (D3):** Grabación "Estresada - Tarea Alta"
   - *Nota:* Para IVA, las fuentes (componentes) deben estar correlacionadas/dependientes a través de los datasets.

2. **Pregunta:** Aplicando IVA, ¿es posible descomponer las señales en vectores fuente que separen:
   - **"Identidad vocal"** del hablante (componente común en todas las vistas)
   - **"Firma del estrés"** (componente que modula la voz consistentemente en vistas D2 y D3)

3. **Análisis:**
   - Interpretar vectores fuente resultantes.
   - ¿Se aisló un componente que represente clara y unívocamente la modulación fisiológica inducida por estrés?
   - Comparación con ICA simple aplicado a cada dataset por separado.

## Entregables

- Código Python/R/Julia con implementación de ICA e IVA sobre audio.
- Tablas, gráficos o indicadores de calidad de separación (SIR).
- Interpretación de resultados y comparación ICA vs. IVA.
