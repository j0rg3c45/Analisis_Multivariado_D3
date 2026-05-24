# Situación 1: Estructuras Latentes en Viajes de NYC (Yellow, Green, FHV, HVFHV)

**Rol:** Científico de Datos  
**Institución:** Universidad Autónoma de Occidente (UAO-Virtual)  
**Dataset:** Registros históricos NYC Taxi (Ene–Abr 2025, Ene–Abr 2026)

---

## 1. Enfoque Metodológico

### 1.1 Integración de Datos

Se integran cuatro servicios de transporte en formato PARQUET:

| Servicio | Archivo | Columnas clave |
|----------|---------|----------------|
| **Yellow Taxi** | `yellow_tripdata_*.parquet` | `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `passenger_count`, `trip_distance`, `fare_amount`, `tip_amount` |
| **Green Taxi** | `green_tripdata_*.parquet` | `lpep_pickup_datetime`, `lpep_dropoff_datetime`, `passenger_count`, `trip_distance`, `fare_amount`, `tip_amount` |
| **FHV** | `fhv_tripdata_*.parquet` | `pickup_datetime`, `dropOff_datetime`, `trip_distance` (no disponible), `tips` (no disponible) |
| **HVFHV** | `fhvhv_tripdata_*.parquet` | `pickup_datetime`, `dropoff_datetime`, `trip_miles`, `base_passenger_fare`, `tips` |

### 1.2 Estandarización de Variables

Se homologan las siguientes variables:

| Variable | Yellow | Green | FHV | HVFHV |
|----------|--------|-------|-----|-------|
| `trip_distance` | `trip_distance` | `trip_distance` | `NA` (imputar) | `trip_miles` |
| `duration_minutes` | `tpep_dropoff_datetime - tpep_pickup_datetime` | `lpep_dropoff_datetime - lpep_pickup_datetime` | `dropOff_datetime - pickup_datetime` | `dropoff_datetime - pickup_datetime` |
| `passenger_count` | `passenger_count` | `passenger_count` | `1` (imputar) | `NA` (imputar 1) |
| `fare_amount` | `fare_amount` | `fare_amount` | `NA` | `base_passenger_fare` |
| `tip_amount` | `tip_amount` | `tip_amount` | `0` (imputar) | `tips` |
| `has_tip` | `tip_amount > 0` | `tip_amount > 0` | `False` | `tips > 0` |

### 1.3 Taxonomía de Reducción de Dimensionalidad

| Método | Principio | Ventaja | Desventaja |
|--------|-----------|---------|------------|
| **PCA** | Maximiza varianza global en componentes ortogonales | Simple, rápido, no paramétrico | Solo captura relaciones lineales |
| **Sparse PCA** | Penalización L1 en cargas | Alta interpretabilidad (coeficientes cero) | Mayor costo computacional |
| **Kernel PCA** | Mapeo no lineal (RBF) | Captura relaciones no lineales | Difícil interpretación de componentes |
| **ICA** | Maximiza no-gaussianidad (curtosis) | Aísla fuentes independientes | Sensible a inicialización |
| **IVA** | Extensión multivista de ICA | Preserva dependencias entre vistas | Requiere implementación especializada |

---

## 2. Implementación en Python

### 2.1 Librerías

```python
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.decomposition import PCA, SparsePCA, KernelPCA, FastICA, FactorAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score

import scipy.io.wavfile as wav
from scipy.signal import convolve

sns.set_theme(style='whitegrid', palette='viridis')
plt.rcParams['figure.figsize'] = (12, 6)

RANDOM_STATE = 42
SAMPLE_SIZE = 100_000  # Muestra balanceada por eficiencia
```

### 2.2 Carga y Preprocesamiento

```python
def load_yellow(path):
    """Carga y estandariza Yellow Taxi."""
    df = pq.read_table(path).to_pandas()
    df = df.rename(columns={
        'tpep_pickup_datetime': 'pickup_datetime',
        'tpep_dropoff_datetime': 'dropoff_datetime'
    })
    df['duration_minutes'] = (
        pd.to_datetime(df['dropoff_datetime']) -
        pd.to_datetime(df['pickup_datetime'])
    ).dt.total_seconds() / 60.0
    df['trip_distance'] = df['trip_distance'].astype(float)
    df['fare_amount'] = df['fare_amount'].astype(float)
    df['tip_amount'] = df['tip_amount'].astype(float)
    df['passenger_count'] = df['passenger_count'].fillna(1).astype(int)
    df['has_tip'] = (df['tip_amount'] > 0).astype(int)
    df['service_type'] = 'yellow'
    return df[['trip_distance', 'duration_minutes', 'passenger_count',
               'fare_amount', 'tip_amount', 'has_tip', 'service_type',
               'PULocationID', 'DOLocationID']]


def load_green(path):
    """Carga y estandariza Green Taxi."""
    df = pq.read_table(path).to_pandas()
    df = df.rename(columns={
        'lpep_pickup_datetime': 'pickup_datetime',
        'lpep_dropoff_datetime': 'dropoff_datetime'
    })
    df['duration_minutes'] = (
        pd.to_datetime(df['dropoff_datetime']) -
        pd.to_datetime(df['pickup_datetime'])
    ).dt.total_seconds() / 60.0
    df['trip_distance'] = df['trip_distance'].astype(float)
    df['fare_amount'] = df['fare_amount'].astype(float)
    df['tip_amount'] = df['tip_amount'].astype(float)
    df['passenger_count'] = df['passenger_count'].fillna(1).astype(int)
    df['has_tip'] = (df['tip_amount'] > 0).astype(int)
    df['service_type'] = 'green'
    return df[['trip_distance', 'duration_minutes', 'passenger_count',
               'fare_amount', 'tip_amount', 'has_tip', 'service_type',
               'PULocationID', 'DOLocationID']]


def load_fhv(path):
    """Carga y estandariza FHV (sin distancia, tarifa ni propina)."""
    df = pq.read_table(path).to_pandas()
    df = df.rename(columns={
        'pickup_datetime': 'pickup_datetime',
        'dropOff_datetime': 'dropoff_datetime'
    })
    df['duration_minutes'] = (
        pd.to_datetime(df['dropoff_datetime']) -
        pd.to_datetime(df['pickup_datetime'])
    ).dt.total_seconds() / 60.0
    df['trip_distance'] = np.nan
    df['fare_amount'] = np.nan
    df['tip_amount'] = 0.0
    df['passenger_count'] = 1
    df['has_tip'] = 0
    df['service_type'] = 'fhv'
    return df[['trip_distance', 'duration_minutes', 'passenger_count',
               'fare_amount', 'tip_amount', 'has_tip', 'service_type',
               'PUlocationID', 'DOlocationID']]


def load_fhvhv(path):
    """Carga y estandariza HVFHV (High Volume FHV)."""
    df = pq.read_table(path).to_pandas()
    df['duration_minutes'] = (
        pd.to_datetime(df['dropoff_datetime']) -
        pd.to_datetime(df['pickup_datetime'])
    ).dt.total_seconds() / 60.0
    df['trip_distance'] = df['trip_miles'].astype(float)
    df['fare_amount'] = df['base_passenger_fare'].astype(float)
    df['tip_amount'] = df['tips'].astype(float)
    df['passenger_count'] = 1  # No disponible en HVFHV
    df['has_tip'] = (df['tip_amount'] > 0).astype(int)
    df['service_type'] = 'fhvhv'
    return df[['trip_distance', 'duration_minutes', 'passenger_count',
               'fare_amount', 'tip_amount', 'has_tip', 'service_type',
               'PULocationID', 'DOLocationID']]
```

### 2.3 Pipeline de Integración

```python
def integrate_all_taxi_data(data_dir, sample_per_service=SAMPLE_SIZE):
    """Integra los 4 servicios de taxi con muestreo balanceado."""
    years_months = [
        ('anio_2025', ['January', 'February', 'March', 'April']),
        ('anio_2026', ['January', 'February', 'March', 'April'])
    ]

    loaders = {
        'yellow': load_yellow,
        'green': load_green,
        'fhv': load_fhv,
        'fhvhv': load_fhvhv
    }

    all_dfs = []
    for year, months in years_months:
        for month in months:
            month_dir = os.path.join(data_dir, year, month)
            for service, loader in loaders.items():
                fname = f'{service}_tripdata_{year[-4:]}-{month[:3] if len(month) > 3 else {"January": "01", "February": "02", "March": "03", "April": "04"}[month]}.parquet'
                fname = f'{service}_tripdata_{year[-4:]}-{month[:3] if len(month) > 3 else month}.parquet'
                # Construcción correcta del nombre
                month_num = {
                    'January': '01', 'February': '02',
                    'March': '03', 'April': '04'
                }[month]
                fname = f'{service}_tripdata_{year[-4:]}-{month_num}.parquet'
                fpath = os.path.join(month_dir, fname)
                if not os.path.exists(fpath):
                    print(f'  [SKIP] {fname} no existe')
                    continue
                try:
                    df = loader(fpath)
                    df['year'] = int(year[-4:])
                    df['month'] = int(month_num)
                    all_dfs.append(df)
                    print(f'  [OK] {fname}: {len(df)} registros')
                except Exception as e:
                    print(f'  [FAIL] {fname}: {e}')

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f'\nTotal integrado: {len(df_all):,} registros')

    return df_all


def clean_outliers(df):
    """Elimina outliers flagrantes y filtra datos inválidos."""
    initial = len(df)
    df = df.dropna(subset=['duration_minutes', 'trip_distance', 'fare_amount'])
    df = df[df['duration_minutes'] > 0]
    df = df[df['duration_minutes'] <= 300]  # Max 5 horas
    df = df[df['trip_distance'] > 0]
    df = df[df['trip_distance'] <= 500]  # Max 500 millas
    df = df[df['fare_amount'] > 0]
    df = df[df['fare_amount'] <= 1000]
    df = df[df['passenger_count'] > 0]
    df = df[df['passenger_count'] <= 9]
    print(f'Outliers eliminados: {initial - len(df)} ({100*(1-len(df)/initial):.1f}%)')
    return df
```

### 2.4 Extracción de Componentes Latentes

```python
def extract_latent_components(X, n_components=3):
    """Aplica múltiples métodos de extracción de componentes latentes."""
    results = {}

    # 1. PCA Tradicional
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    results['pca'] = {
        'scores': X_pca,
        'components': pca.components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'model': pca
    }
    print(f'PCA: Varianza explicada = {pca.explained_variance_ratio_.sum():.3f}')

    # 2. Sparse PCA (Disperso con L1)
    spca = SparsePCA(n_components=n_components, alpha=1.0,
                     random_state=RANDOM_STATE)
    X_spca = spca.fit_transform(X)
    results['sparse_pca'] = {
        'scores': X_spca,
        'components': spca.components_,
        'model': spca
    }
    print('Sparse PCA: Completo')

    # 3. Kernel PCA (RBF)
    kpca = KernelPCA(n_components=n_components, kernel='rbf',
                     gamma=0.1, random_state=RANDOM_STATE)
    X_kpca = kpca.fit_transform(X)
    results['kernel_pca'] = {
        'scores': X_kpca,
        'components': kpca.alphas_ if hasattr(kpca, 'alphas_') else None,
        'model': kpca
    }
    print('Kernel PCA (RBF): Completo')

    # 4. ICA (FastICA)
    ica = FastICA(n_components=n_components, whiten='unit-variance',
                  random_state=RANDOM_STATE, max_iter=1000)
    X_ica = ica.fit_transform(X)
    results['ica'] = {
        'scores': X_ica,
        'components': ica.components_,
        'mixing_matrix': ica.mixing_,
        'model': ica
    }
    print('ICA: Completo')

    # 5. IVA - Independent Vector Analysis (aproximación multivista)
    # Simulamos 2 vistas dividiendo las características
    n_features = X.shape[1]
    mid = n_features // 2
    X_view1 = X[:, :mid]
    X_view2 = X[:, mid:]

    ica_v1 = FastICA(n_components=min(n_components, mid),
                     whiten='unit-variance', random_state=RANDOM_STATE)
    ica_v2 = FastICA(n_components=min(n_components, n_features - mid),
                     whiten='unit-variance', random_state=RANDOM_STATE)

    S1 = ica_v1.fit_transform(X_view1)
    S2 = ica_v2.fit_transform(X_view2)

    # Alineación por correlación entre vistas
    corr_matrix = np.abs(np.corrcoef(S1.T, S2.T)[:n_components, n_components:])
    alignment = np.argmax(corr_matrix, axis=1)

    results['iva'] = {
        'scores_v1': S1,
        'scores_v2': S2,
        'components_v1': ica_v1.components_,
        'components_v2': ica_v2.components_,
        'alignment': alignment,
        'cross_correlation': corr_matrix,
        'model_v1': ica_v1,
        'model_v2': ica_v2
    }
    print('IVA (multivista): Completo')

    return results


def interpret_components(results, feature_names):
    """Interpreta los componentes de cada método."""
    interpretations = {}

    for method, res in results.items():
        if method == 'iva':
            continue
        comps = res['components']
        n_comp = comps.shape[0]
        interpretations[method] = []
        for i in range(n_comp):
            loadings = pd.Series(comps[i], index=feature_names)
            top_pos = loadings.nlargest(3)
            top_neg = loadings.nsmallest(3)
            interpretations[method].append({
                'component': i + 1,
                'top_positive': dict(top_pos),
                'top_negative': dict(top_neg),
                'sparsity': (np.abs(loadings) < 0.01).mean()
            })

    return interpretations
```

### 2.5 Análisis Factorial Exploratorio (EFA)

```python
def exploratory_factor_analysis(X, n_factors=3, rotation='varimax'):
    """Aplica Análisis Factorial Exploratorio."""
    fa = FactorAnalysis(n_components=n_factors, rotation=rotation,
                        random_state=RANDOM_STATE)
    X_fa = fa.fit_transform(X)
    loadings = fa.components_.T  # p x k matrix

    # Comunalidades
    communalities = np.sum(loadings ** 2, axis=1)

    # Varianza explicada por cada factor
    variance_explained = np.var(X_fa, axis=0)
    prop_variance = variance_explained / np.sum(variance_explained)

    return {
        'scores': X_fa,
        'loadings': loadings,
        'communalities': communalities,
        'prop_variance': prop_variance,
        'model': fa
    }
```

### 2.6 Clustering sobre Componentes Latentes

```python
def cluster_analysis(X_latent, method='kmeans', n_clusters=4):
    """Aplica clustering sobre componentes latentes."""
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE,
                       n_init=10)
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters,
                                random_state=RANDOM_STATE)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError(f'Método {method} no soportado')

    labels = model.fit_predict(X_latent)

    # Métricas de validación
    metrics = {
        'silhouette': silhouette_score(X_latent, labels),
        'calinski_harabasz': calinski_harabasz_score(X_latent, labels),
        'davies_bouldin': davies_bouldin_score(X_latent, labels)
    }

    return {'labels': labels, 'model': model, 'metrics': metrics}


def optimal_n_clusters(X_latent, max_k=10):
    """Determina el número óptimo de clusters usando codo y silhouette."""
    inertias = []
    sil_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_latent)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_latent, labels))

    return {
        'k_range': list(K_range),
        'inertias': inertias,
        'silhouette_scores': sil_scores,
        'optimal_k': K_range[np.argmax(sil_scores)]
    }
```

### 2.7 Validez Predictiva

```python
def predictive_validity(X_latent, df):
    """Evalúa la capacidad predictiva de los componentes sobre las 3 variables objetivo."""
    results = {}

    targets = {
        'has_tip': 'binaria (propina)',
        'passenger_count': 'conteo (pasajeros)',
        'tip_amount': 'continua (monto propina)'
    }

    for target, desc in targets.items():
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(
            X_latent, y, test_size=0.3, random_state=RANDOM_STATE
        )

        if target == 'has_tip':
            model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            model.fit(X_train, y_train)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_pred_prob)
            metric = 'AUC-ROC'
        elif target == 'passenger_count':
            model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = (y_pred == y_test).mean()
            metric = 'Accuracy'
        else:
            mask = y > 0
            if mask.sum() < 100:
                score = np.nan
                metric = 'R² (insuf. datos)'
            else:
                X_filt = X_latent[mask]
                y_filt = y[mask]
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_filt, y_filt, test_size=0.3, random_state=RANDOM_STATE
                )
                model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                score = r2_score(y_te, y_pred)
                metric = 'R²'

        results[target] = {
            'metric': metric,
            'score': score,
            'description': desc
        }

    return results
```

### 2.8 Visualización

```python
def plot_explained_variance(pca_result, title='Varianza Explicada - PCA'):
    """Gráfico de varianza explicada."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    evr = pca_result['explained_variance_ratio']
    cumsum = np.cumsum(evr)

    axes[0].bar(range(1, len(evr) + 1), evr, alpha=0.7, color='steelblue',
                edgecolor='black')
    axes[0].set_xlabel('Componente Principal')
    axes[0].set_ylabel('Varianza Explicada (Proporción)')
    axes[0].set_title('Varianza Explicada por Componente')
    axes[0].axhline(y=0.1, color='red', linestyle='--', alpha=0.5)

    axes[1].plot(range(1, len(evr) + 1), cumsum, 'o-', color='darkred',
                 linewidth=2, markersize=8)
    axes[1].axhline(y=0.7, color='gray', linestyle='--', alpha=0.5,
                    label='70% umbral')
    axes[1].set_xlabel('Número de Componentes')
    axes[1].set_ylabel('Varianza Explicada Acumulada')
    axes[1].set_title('Varianza Explicada Acumulada')
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('varianza_explicada.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_loadings_heatmap(loadings, feature_names, method_name):
    """Mapa de calor de cargas factoriales."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        loadings.T, annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, xticklabels=feature_names,
        yticklabels=[f'Comp {i+1}' for i in range(loadings.shape[0])],
        linewidths=0.5, ax=ax, vmin=-1, vmax=1
    )
    ax.set_title(f'Cargas Factoriales - {method_name}', fontweight='bold')
    ax.set_xlabel('Variables')
    ax.set_ylabel('Componentes')
    plt.tight_layout()
    plt.savefig(f'loadings_{method_name.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_clusters_2d(X_latent, labels, method_name, title_extra=''):
    """Visualización de clusters en 2D (primeras 2 componentes)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        X_latent[:, 0], X_latent[:, 1],
        c=labels, cmap='viridis', alpha=0.6,
        edgecolors='black', linewidth=0.3, s=20
    )
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')
    ax.set_title(f'Clusters sobre {method_name} - {title_extra}',
                 fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    plt.savefig(f'clusters_{method_name.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_method_comparison(results_df):
    """Gráfico comparativo de métodos."""
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(results_df))
    width = 0.2
    metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    for i, metric in enumerate(metrics):
        if metric in results_df.columns:
            values = results_df[metric].values
            # Normalizar para visualización
            if metric == 'davies_bouldin':
                values = 1 / (values + 1e-10)  # Invertir (menor es mejor)
            vmin, vmax = values.min(), values.max()
            if vmax > vmin:
                values = (values - vmin) / (vmax - vmin)
            bars = ax.bar(x + i * width, values, width, label=metric,
                          color=colors[i], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df.index)
    ax.set_ylabel('Puntaje Normalizado')
    ax.set_title('Comparación de Métodos de Clustering sobre Espacios Latentes')
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparacion_metodos.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 3. Ejecución Completa

```python
def main():
    # Configuración
    data_dir = r'C:\Users\Jorge\Documents\01.MAESTRIA_IA\02_SEMESTRE_02-2026\03_ANALISIS_MULTIVARIADO\03_DESAFIO_03\data\data_situacion_01'

    print('=' * 60)
    print('SITUACIÓN 1: Estructuras Latentes en Viajes NYC')
    print('=' * 60)

    # 1. Integración de Datos
    print('\n[1] Integrando datos de los 4 servicios de taxi...')
    df_all = integrate_all_taxi_data(data_dir)

    # 2. Limpieza
    print('\n[2] Limpiando outliers...')
    df_all = clean_outliers(df_all)
    print(f'  Dataset final: {len(df_all):,} registros')

    # 3. Ingeniería de características
    print('\n[3] Preparando variables para modelado...')
    features = ['trip_distance', 'duration_minutes', 'fare_amount',
                'passenger_count']

    # Filtrar filas con datos completos para features seleccionados
    df_model = df_all.dropna(subset=features).copy()
    print(f'  Registros con datos completos: {len(df_model):,}')

    # Muestreo balanceado
    if len(df_model) > SAMPLE_SIZE:
        df_sample = df_model.groupby('service_type', group_keys=False).apply(
            lambda x: x.sample(min(len(x), SAMPLE_SIZE // 4),
                               random_state=RANDOM_STATE)
        ).reset_index(drop=True)
    else:
        df_sample = df_model

    print(f'  Muestra balanceada: {len(df_sample):,} registros')
    print(f'  Distribución por servicio:')
    print(df_sample['service_type'].value_counts())

    # 4. Escalado
    print('\n[4] Escalando características...')
    scaler = StandardScaler()
    X = scaler.fit_transform(df_sample[features])

    # 5. Extracción de Componentes Latentes
    print('\n[5] Extrayendo componentes latentes...')
    n_components = 3
    latent_results = extract_latent_components(X, n_components=n_components)

    # Interpretación de componentes
    interpretations = interpret_components(latent_results, features)
    for method, comps in interpretations.items():
        print(f'\n{method.upper()}:')
        for comp in comps:
            print(f'  Comp {comp["component"]}: '
                  f'Top+ = {comp["top_positive"]}, '
                  f'Sparsity = {comp["sparsity"]:.2%}')

    # 6. Análisis Factorial Exploratorio
    print('\n[6] Análisis Factorial Exploratorio...')
    efa = exploratory_factor_analysis(X, n_factors=n_components)
    print(f'  Varianza explicada por factor: {efa["prop_variance"]}')

    # 7. Clustering
    print('\n[7] Clustering sobre componentes latentes...')
    clustering_results = {}

    for method_name, res in latent_results.items():
        if method_name == 'iva':
            scores = res['scores_v1']
        else:
            scores = res['scores']

        # Determinar K óptimo
        opt = optimal_n_clusters(scores, max_k=8)
        optimal_k = opt['optimal_k']

        # Aplicar clustering
        clust = cluster_analysis(scores, method='kmeans',
                                 n_clusters=optimal_k)
        clustering_results[method_name] = {
            'labels': clust['labels'],
            'metrics': clust['metrics'],
            'optimal_k': optimal_k,
            'silhouette_scores': opt['silhouette_scores']
        }

        print(f'  {method_name}: K={optimal_k}, '
              f'Silhouette={clust["metrics"]["silhouette"]:.3f}, '
              f'CH={clust["metrics"]["calinski_harabasz"]:.0f}, '
              f'DB={clust["metrics"]["davies_bouldin"]:.3f}')

    # 8. Validez Predictiva
    print('\n[8] Evaluando validez predictiva...')
    predictive_results = {}

    for method_name, res in latent_results.items():
        if method_name == 'iva':
            scores = res['scores_v1']
        else:
            scores = res['scores']

        pred = predictive_validity(scores, df_sample)
        predictive_results[method_name] = pred

        print(f'\n  {method_name}:')
        for target, info in pred.items():
            print(f'    {target} ({info["description"]}): '
                  f'{info["metric"]} = {info["score"]:.4f}')

    # 9. Resumen Comparativo
    print('\n' + '=' * 60)
    print('RESUMEN COMPARATIVO DE MÉTODOS')
    print('=' * 60)

    comparison = pd.DataFrame({
        method: {
            'silhouette': clustering_results[method]['metrics']['silhouette'],
            'calinski_harabasz': clustering_results[method]['metrics']['calinski_harabasz'],
            'davies_bouldin': clustering_results[method]['metrics']['davies_bouldin'],
            'k_optimo': clustering_results[method]['optimal_k'],
            'auc_propina': predictive_results[method]['has_tip']['score'],
            'r2_tip_amount': predictive_results[method]['tip_amount']['score']
        }
        for method in clustering_results
    }).T.round(4)
    print(comparison.sort_values('silhouette', ascending=False))

    # Señalar el mejor modelo
    best_method = comparison['silhouette'].idxmax()
    print(f'\n--- Mejor modelo: {best_method} '
          f'(Silhouette={comparison.loc[best_method, "silhouette"]}) ---')

    return {
        'df': df_all,
        'df_sample': df_sample,
        'X_scaled': X,
        'latent_results': latent_results,
        'interpretations': interpretations,
        'efa': efa,
        'clustering_results': clustering_results,
        'predictive_results': predictive_results,
        'comparison': comparison,
        'best_method': best_method,
        'features': features,
        'scaler': scaler
    }


if __name__ == '__main__':
    results = main()
```

---

## 4. Interpretación de Resultados

### 4.1 Constructos Latentes Esperados

Basado en la literatura de transporte urbano, se espera identificar los siguientes factores:

| Factor Latente | Variables Asociadas | Interpretación |
|----------------|-------------------|----------------|
| **Eficiencia del Viaje** | `duration_minutes` (–), `trip_distance` (–) | Viajes rápidos y cortos (alta eficiencia) |
| **Costo-Distancia** | `fare_amount` (+), `trip_distance` (+), `duration_minutes` (+) | Viajes largos y costosos |
| **Demanda del Cliente** | `passenger_count` (+), `has_tip` (+) | Viajes con mayor ocupación y propina |
| **Generosidad (Propina)** | `tip_amount` (+), `fare_amount` (+) | Monto de propina correlacionado con tarifa |

### 4.2 Criterios de Evaluación

| Criterio | Métrica | Método esperado superior |
|----------|---------|-------------------------|
| **Interpretabilidad** | Cargas esparsas (Sparse PCA) | Sparse PCA |
| **Parsimonia** | Número de coeficientes no nulos | Sparse PCA > PCA > ICA |
| **Segmentación** | Silhouette Score | Kernel PCA (no lineal) |
| **Validez Predictiva** | AUC-ROC (has_tip), R² (tip_amount) | ICA / IVA |

### 4.3 Preguntas Guía para el Análisis

1. **¿Cuál método produce componentes más interpretables?**
   - Sparse PCA fuerza cargas a cero, facilitando la interpretación.
   - PCA tradicional produce cargas densas (todas las variables contribuyen).
   - ICA separa fuentes independientes que pueden corresponder a procesos operacionales reales.

2. **¿Qué retiene Kernel PCA que los métodos lineales no capturan?**
   - Relaciones no lineales entre distancia, duración y tarifa (ej. tarifas dinámicas).

3. **¿IVA ofrece ventaja sobre ICA individual?**
   - IVA preserva la correlación entre vistas (ej. comportamiento similar en diferentes periodos), lo que puede estabilizar la estimación de componentes.

4. **¿Los clusters son estables entre métodos?**
   - Comparar el Adjusted Rand Index (ARI) entre asignaciones de clusters de diferentes espacios latentes para evaluar consistencia.

5. **¿Qué componentes predicen mejor la propina?**
   - Un componente de "generosidad" debería tener alta capacidad predictiva sobre `has_tip` y `tip_amount`.

---

## 5. Notas Técnicas

### 5.1 Optimización con `uv` y `Warp`

```bash
# Crear entorno virtual
uv venv

# Activar (PowerShell)
.venv\Scripts\Activate.ps1

# Instalar dependencias
uv pip install pyarrow pandas numpy scikit-learn scipy matplotlib seaborn

# Ejecutar el script
uv run python situacion_01.py
```

### 5.2 Manejo de Memoria

- Los archivos PARQUET de NYC Taxi pueden exceder 20M de registros/mes.
- Se aplica muestreo estratificado por servicio (`SAMPLE_SIZE=100,000`).
- Para análisis completo sin muestreo, usar lectura perezosa con `pq.ParquetFile` y procesamiento por lotes.

### 5.3 IVA - Implementación

Dado que IVA no está disponible en scikit-learn, se implementa una aproximación:
1. Dividir las características en dos vistas complementarias.
2. Aplicar FastICA independientemente en cada vista.
3. Alinear componentes por máxima correlación cruzada.

Para una implementación rigurosa de IVA, se recomienda:
- **Python:** `iva` package (`pip install iva`) o `groupica`.
- **Matlab:** Toolbox `IVA` del grupo de Tülay Adalı (UMBC).
- **R:** `iva` package en CRAN.
