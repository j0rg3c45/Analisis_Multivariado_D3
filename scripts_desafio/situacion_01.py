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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, r2_score

sns.set_theme(style='whitegrid', palette='viridis')
plt.rcParams['figure.figsize'] = (12, 6)

RANDOM_STATE = 42
SAMPLE_SIZE = 100_000  # Muestra balanceada por eficiencia

# --- Loaders ---

def load_yellow(path, n_sample=10000):
    """Carga y estandariza Yellow Taxi."""
    cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 
            'trip_distance', 'fare_amount', 'tip_amount', 'PULocationID', 'DOLocationID']
    
    pfile = pq.ParquetFile(path)
    if pfile.metadata.num_rows > n_sample * 2:
        # Sample by reading a chunk or specific row groups
        df = pfile.read_row_group(0, columns=cols).to_pandas()
        if len(df) > n_sample:
            df = df.sample(n_sample, random_state=RANDOM_STATE)
    else:
        df = pq.read_table(path, columns=cols).to_pandas()
        if len(df) > n_sample:
            df = df.sample(n_sample, random_state=RANDOM_STATE)

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


def load_green(path, n_sample=10000):
    """Carga y estandariza Green Taxi."""
    cols = ['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'passenger_count', 
            'trip_distance', 'fare_amount', 'tip_amount', 'PULocationID', 'DOLocationID']
    
    pfile = pq.ParquetFile(path)
    if pfile.metadata.num_rows > n_sample * 2:
        df = pfile.read_row_group(0, columns=cols).to_pandas()
        if len(df) > n_sample:
            df = df.sample(n_sample, random_state=RANDOM_STATE)
    else:
        df = pq.read_table(path, columns=cols).to_pandas()
        if len(df) > n_sample:
            df = df.sample(n_sample, random_state=RANDOM_STATE)

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


def load_fhv(path, n_sample=10000):
    """Carga y estandariza FHV (sin distancia, tarifa ni propina)."""
    # Find available columns first to avoid error
    pfile = pq.ParquetFile(path)
    all_cols = pfile.schema.names
    cols = ['pickup_datetime', 'dropOff_datetime']
    loc_cols = [c for c in all_cols if 'locationid' in c.lower()]
    cols.extend(loc_cols)

    if pfile.metadata.num_rows > n_sample * 2:
        df = pfile.read_row_group(0, columns=cols).to_pandas()
        if len(df) > n_sample:
            df = df.sample(n_sample, random_state=RANDOM_STATE)
    else:
        df = pq.read_table(path, columns=cols).to_pandas()
        if len(df) > n_sample:
            df = df.sample(n_sample, random_state=RANDOM_STATE)

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
    
    rename_dict = {}
    for col in loc_cols:
        if 'pu' in col.lower(): rename_dict[col] = 'PULocationID'
        if 'do' in col.lower(): rename_dict[col] = 'DOLocationID'
    df = df.rename(columns=rename_dict)
    
    return df[['trip_distance', 'duration_minutes', 'passenger_count',
               'fare_amount', 'tip_amount', 'has_tip', 'service_type',
               'PULocationID', 'DOLocationID']]


def load_fhvhv(path, n_sample=10000):
    """Carga y estandariza HVFHV (High Volume FHV)."""
    cols = ['pickup_datetime', 'dropoff_datetime', 'trip_miles', 'base_passenger_fare', 'tips', 'PULocationID', 'DOLocationID']
    
    pfile = pq.ParquetFile(path)
    if pfile.metadata.num_rows > n_sample * 2:
        # HVFHV files are usually the largest. Read only first row group.
        df = pfile.read_row_group(0, columns=cols).to_pandas()
        if len(df) > n_sample:
            df = df.sample(n_sample, random_state=RANDOM_STATE)
    else:
        df = pq.read_table(path, columns=cols).to_pandas()
        if len(df) > n_sample:
            df = df.sample(n_sample, random_state=RANDOM_STATE)

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

# --- Pipeline ---

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

    # Total files expected: 4 services * 4 months * 2 years = 32
    # To get SAMPLE_SIZE in total, we need ~3125 per file
    n_per_file = sample_per_service // 32

    all_dfs = []
    for year, months in years_months:
        for month in months:
            month_dir = os.path.join(data_dir, year, month)
            for service, loader in loaders.items():
                month_num = {
                    'January': '01', 'February': '02',
                    'March': '03', 'April': '04'
                }[month]
                fname = f'{service}_tripdata_{year[-4:]}-{month_num}.parquet'
                fpath = os.path.join(month_dir, fname)
                if not os.path.exists(fpath):
                    continue
                try:
                    df = loader(fpath, n_sample=n_per_file)
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
    # Para FHV no tenemos trip_distance ni fare_amount, así que no podemos filtrar por ellos si queremos conservarlos
    # Pero el análisis latente requiere estas variables. El MD dice:
    # "Filtrar filas con datos completos para features seleccionados"
    
    # Solo aplicamos filtros a lo que no sea NaN
    mask = pd.Series(True, index=df.index)
    
    if 'duration_minutes' in df.columns:
        mask &= (df['duration_minutes'] > 0) & (df['duration_minutes'] <= 300)
    
    # Para las otras variables, el MD sugiere filtrarlas. 
    # Si queremos incluir FHV, tendremos que imputar o aceptar NaNs.
    # El MD en 3. "main" dice: df_model = df_all.dropna(subset=features).copy()
    # features = ['trip_distance', 'duration_minutes', 'fare_amount', 'passenger_count']
    # Esto eliminaría FHV porque trip_distance y fare_amount son NaN.
    
    # REVISIÓN: El MD dice para FHV: 'trip_distance': np.nan, 'fare_amount': np.nan
    # Pero luego dice que se filtran. Esto es una contradicción si se quiere analizar FHV.
    # Sin embargo, seguiré el MD que dice "dropna(subset=features)".
    
    df = df[mask]
    
    print(f'Outliers básicos eliminados: {initial - len(df)} ({100*(1-len(df)/initial):.1f}%)')
    return df

# --- Latent Extraction ---

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

    # 3. Kernel PCA (RBF) - Usamos una muestra más pequeña si es necesario por costo computacional
    # Kernel PCA es O(N^3). Con 100k es imposible.
    kpca_sample_size = min(5000, X.shape[0])
    indices = np.random.choice(X.shape[0], kpca_sample_size, replace=False)
    X_kpca_sub = X[indices]
    
    kpca = KernelPCA(n_components=n_components, kernel='rbf',
                     gamma=0.1, random_state=RANDOM_STATE, fit_inverse_transform=False)
    X_kpca = kpca.fit_transform(X_kpca_sub)
    
    # Para el resto de X, transformamos (si cabe en memoria)
    # X_kpca_full = kpca.transform(X) # Esto también es lento. 
    # Por ahora solo guardamos el subconjunto para visualización/análisis rápido.
    results['kernel_pca'] = {
        'scores': X_kpca,
        'indices': indices,
        'model': kpca
    }
    print(f'Kernel PCA (RBF) en muestra de {kpca_sample_size}: Completo')

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
    corr_matrix = np.abs(np.corrcoef(S1.T, S2.T)[:S1.shape[1], S1.shape[1]:])
    alignment = np.argmax(corr_matrix, axis=1) if corr_matrix.size > 0 else []

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
        if method in ['iva', 'kernel_pca']:
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

# --- EFA ---

def exploratory_factor_analysis(X, n_factors=3, rotation='varimax'):
    """Aplica Análisis Factorial Exploratorio."""
    fa = FactorAnalysis(n_components=n_factors, rotation=rotation,
                        random_state=RANDOM_STATE)
    X_fa = fa.fit_transform(X)
    loadings = fa.components_.T  # p x k matrix

    communalities = np.sum(loadings ** 2, axis=1)
    variance_explained = np.var(X_fa, axis=0)
    prop_variance = variance_explained / np.sum(variance_explained)

    return {
        'scores': X_fa,
        'loadings': loadings,
        'communalities': communalities,
        'prop_variance': prop_variance,
        'model': fa
    }

# --- Clustering ---

def cluster_analysis(X_latent, method='kmeans', n_clusters=4):
    """Aplica clustering sobre componentes latentes."""
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE,
                       n_init=10)
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters,
                                random_state=RANDOM_STATE)
    elif method == 'agglomerative':
        # Agglomerative es lento para N grande
        if len(X_latent) > 10000:
            X_latent = X_latent[:10000]
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError(f'Método {method} no soportado')

    labels = model.fit_predict(X_latent)

    metrics = {
        'silhouette': silhouette_score(X_latent, labels),
        'calinski_harabasz': calinski_harabasz_score(X_latent, labels),
        'davies_bouldin': davies_bouldin_score(X_latent, labels)
    }

    return {'labels': labels, 'model': model, 'metrics': metrics}


def optimal_n_clusters(X_latent, max_k=10):
    """Determina el número óptimo de clusters usando codo y silhouette."""
    # Reducimos X si es muy grande para velocidad
    if len(X_latent) > 5000:
        X_sub = X_latent[np.random.choice(len(X_latent), 5000, replace=False)]
    else:
        X_sub = X_latent

    inertias = []
    sil_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_sub)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_sub, labels))

    return {
        'k_range': list(K_range),
        'inertias': inertias,
        'silhouette_scores': sil_scores,
        'optimal_k': K_range[np.argmax(sil_scores)]
    }

# --- Predictive Validity ---

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
        # Alinear y con X_latent si hubo submuestreo previo
        if len(X_latent) != len(y):
            # Caso especial para Kernel PCA que tiene sus propios índices
            continue

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
            # Simplificar a clasificación de 1 vs más de 1 si es muy complejo, 
            # pero el MD dice RandomForestClassifier
            model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_STATE)
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
                model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=RANDOM_STATE)
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

# --- Visualization ---

def plot_all_results(latent_results, clustering_results, features, comparison):
    """Genera y guarda gráficos clave."""
    out_dir = 'output/situacion_01/'
    # 1. Varianza PCA
    pca_res = latent_results['pca']
    evr = pca_res['explained_variance_ratio']
    cumsum = np.cumsum(evr)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, len(evr) + 1), evr)
    axes[0].set_title('Varianza Explicada por Componente')
    axes[1].plot(range(1, len(evr) + 1), cumsum, 'o-')
    axes[1].set_title('Varianza Acumulada')
    plt.savefig(f'{out_dir}situacion_01_varianza_pca.png')
    plt.close()

    # 2. Heatmap de Cargas PCA
    loadings = latent_results['pca']['components']
    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings, annot=True, cmap='RdBu_r', center=0,
                xticklabels=features, yticklabels=[f'C{i+1}' for i in range(loadings.shape[0])])
    plt.title('Cargas Factoriales PCA')
    plt.savefig(f'{out_dir}situacion_01_loadings_pca.png')
    plt.close()

    # 3. Clusters PCA
    pca_scores = latent_results['pca']['scores']
    labels = clustering_results['pca']['labels']
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_scores[:, 0], pca_scores[:, 1], c=labels, cmap='viridis', s=10, alpha=0.5)
    plt.title('Clusters en Espacio Latente PCA')
    plt.colorbar(label='Cluster')
    plt.savefig(f'{out_dir}situacion_01_clusters_pca.png')
    plt.close()

    # 4. Comparación Métodos
    comparison.plot(kind='bar', figsize=(12, 6))
    plt.title('Comparación de Métodos')
    plt.savefig(f'{out_dir}situacion_01_comparacion_metodos.png')
    plt.close()

# --- Main ---

def main():
    data_dir = r'C:\Users\Jorge\Documents\01.MAESTRIA_IA\02_SEMESTRE_02-2026\03_ANALISIS_MULTIVARIADO\03_DESAFIO_03\data\data_situacion_01'

    print('=' * 60)
    print('SITUACIÓN 1: Estructuras Latentes en Viajes NYC')
    print('=' * 60)

    # 1. Integración
    df_all = integrate_all_taxi_data(data_dir)

    # 2. Limpieza
    df_all = clean_outliers(df_all)

    # 3. Preparación
    features = ['trip_distance', 'duration_minutes', 'fare_amount', 'passenger_count']
    df_model = df_all.dropna(subset=features).copy()
    print(f'Registros con datos completos: {len(df_model):,}')

    if len(df_model) > SAMPLE_SIZE:
        df_sample = df_model.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
    else:
        df_sample = df_model

    print(f'Muestra final: {len(df_sample):,} registros')

    # 4. Escalado
    scaler = StandardScaler()
    X = scaler.fit_transform(df_sample[features])

    # --- [REQUISITO 1: EXTRACCIÓN DE COMPONENTES LATENTES] ---
    # Este bloque aborda la Interpretación y Parsimonia (Criterio 1)
    latent_results = extract_latent_components(X, n_components=3)
    
    # 6. Interpretación de Componentes (Criterio: Interpretatibilidad y Parsimonia)
    # Se identifican constructos como "Eficiencia/Costo" y "Escala del Viaje"
    interpretations = interpret_components(latent_results, features)
    for method, comps in interpretations.items():
        print(f'\n{method.upper()}:')
        for comp in comps:
            print(f'  Comp {comp["component"]}: Top+ = {comp["top_positive"]}')

    # 7. Clustering (Criterio 2: Coherencia de Segmentación)
    # Se evalúa la utilidad operativa de los clústeres identificados
    clustering_results = {}
    for method_name, res in latent_results.items():
        if method_name == 'iva':
            scores = res['scores_v1']
        elif method_name == 'kernel_pca':
            scores = res['scores']
        else:
            scores = res['scores']

        opt = optimal_n_clusters(scores, max_k=6)
        clust = cluster_analysis(scores, n_clusters=opt['optimal_k'])
        clustering_results[method_name] = {
            'labels': clust['labels'],
            'metrics': clust['metrics'],
            'optimal_k': opt['optimal_k']
        }
        print(f'Clustering {method_name}: K={opt["optimal_k"]}, Silhouette={clust["metrics"]["silhouette"]:.3f}')

    # 8. Validez Predictiva (Criterio 3 y Preguntas 1, 2, 3)
    # Evaluación sobre: 1. Probabilidad propina, 2. Pasajeros, 3. Monto propina
    predictive_results = {}
    for method_name, res in latent_results.items():
        if method_name == 'kernel_pca': 
            # Necesitamos alinear df_sample con los índices de Kernel PCA
            scores = res['scores']
            df_sub = df_sample.iloc[res['indices']]
            pred = predictive_validity(scores, df_sub)
        elif method_name == 'iva':
            scores = res['scores_v1']
            pred = predictive_validity(scores, df_sample)
        else:
            scores = res['scores']
            pred = predictive_validity(scores, df_sample)
            
        predictive_results[method_name] = pred

    # 9. Resumen
    comparison = pd.DataFrame({
        method: {
            'silhouette': clustering_results[method]['metrics']['silhouette'],
            'auc_propina': predictive_results[method].get('has_tip', {}).get('score', np.nan),
            'r2_monto': predictive_results[method].get('tip_amount', {}).get('score', np.nan)
        }
        for method in clustering_results
    }).T
    print('\nResumen Comparativo:')
    print(comparison)

    # Visualización
    plot_all_results(latent_results, clustering_results, features, comparison)
    print('\nGráficos guardados.')

if __name__ == '__main__':
    main()
