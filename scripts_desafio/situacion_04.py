import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer

sns.set_theme(style='whitegrid', palette='viridis')

RANDOM_STATE = 42
OUT_DIR = 'output/situacion_04/'

def parse_geo(geo_str):
    try:
        geo = json.loads(geo_str)
        return geo['coordinates']
    except:
        return [np.nan, np.nan]

def load_and_prepare_data(file_path):
    print(f'Cargando datos desde {file_path}...')
    df = pd.read_csv(file_path)
    print(f'Registros: {len(df)}')
    print(f'Columnas: {df.shape[1]}')

    coords = df['.geo'].apply(parse_geo)
    df['lon'] = coords.apply(lambda x: x[0])
    df['lat'] = coords.apply(lambda x: x[1])

    indices = ['evi', 'nbr', 'ndmi', 'ndvi']
    index_data = {}
    for idx in indices:
        cols = sorted([c for c in df.columns if idx in c and c != '.geo' and c != 'system:index'])
        index_data[idx] = {
            'cols': cols,
            'values': df[cols].values
        }
        print(f'  {idx.upper()}: {len(cols)} anos detectados')

    years = [2015 + i for i in range(len(index_data['ndvi']['cols']))]
    print(f'Periodo: {years[0]}-{years[-1]} ({len(years)} anos)')

    return df, index_data, years

def impute_and_scale(index_data):
    X_scaled = {}
    for idx, data in index_data.items():
        X = data['values'].copy()
        if np.isnan(X).any():
            n_nan = np.isnan(X).sum()
            print(f'  {idx.upper()}: {n_nan} NaN imputados ({100*n_nan/X.size:.2f}%)')
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled[idx] = scaler.fit_transform(X)
    return X_scaled

def apply_pca(X_combined, n_components=5):
    print('\n--- PCA ---')
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_combined)
    evr = pca.explained_variance_ratio_
    cumsum = np.cumsum(evr)
    for i, (v, c) in enumerate(zip(evr, cumsum)):
        print(f'  PC{i+1}: var={v:.4f} | acum={c:.4f}')
    print(f'  Varianza total (5 comp): {cumsum[-1]:.4f}')
    return pca, X_pca

def apply_ica(X_combined, n_components=5):
    print('\n--- ICA ---')
    ica = FastICA(n_components=n_components, random_state=RANDOM_STATE, max_iter=1000)
    X_ica = ica.fit_transform(X_combined)
    curtosis = [np.abs(pd.Series(X_ica[:, i]).kurtosis()) for i in range(n_components)]
    for i, k in enumerate(curtosis):
        print(f'  IC{i+1}: curtosis={k:.2f}')
    print(f'  Max curtosis: IC{np.argmax(curtosis)+1} ({max(curtosis):.2f})')
    return ica, X_ica, curtosis

def apply_iva_multivista(index_data, X_scaled, n_comp_iva=5):
    print('\n--- IVA Multivista ---')
    index_pairs = [('ndvi', 'nbr'), ('evi', 'ndmi'), ('ndvi', 'evi')]
    iva_results = {}

    for v1_name, v2_name in index_pairs:
        print(f'\n  Vistas: {v1_name.upper()} <-> {v2_name.upper()}')
        ica_v1 = FastICA(n_components=n_comp_iva, random_state=RANDOM_STATE)
        ica_v2 = FastICA(n_components=n_comp_iva, random_state=RANDOM_STATE)

        S1 = ica_v1.fit_transform(X_scaled[v1_name])
        S2 = ica_v2.fit_transform(X_scaled[v2_name])

        corr_matrix = np.abs(np.corrcoef(S1.T, S2.T)[:n_comp_iva, n_comp_iva:])
        alignment = np.argmax(corr_matrix, axis=1)

        X_iva = np.zeros((S1.shape[0], n_comp_iva))
        for i in range(n_comp_iva):
            X_iva[:, i] = (S1[:, i] + S2[:, alignment[i]]) / 2.0
            print(f'    IVA-{i+1}: {v1_name.upper()}-IC{i+1} <-> {v2_name.upper()}-IC{alignment[i]+1} (corr={corr_matrix[i, alignment[i]]:.4f})')

        # score combinado de alineacion (mean cross-corr)
        alignment_score = np.mean([corr_matrix[i, alignment[i]] for i in range(n_comp_iva)])
        print(f'    Alignment Score: {alignment_score:.4f}')

        iva_results[f'{v1_name}_vs_{v2_name}'] = {
            'X_iva': X_iva,
            'S1': S1, 'S2': S2,
            'alignment': alignment,
            'cross_corr': corr_matrix,
            'alignment_score': alignment_score,
            'model_v1': ica_v1,
            'model_v2': ica_v2
        }

    # Seleccionar el mejor par de vistas
    best_pair = max(iva_results, key=lambda k: iva_results[k]['alignment_score'])
    print(f'\n  Mejor par de vistas: {best_pair} (score={iva_results[best_pair]["alignment_score"]:.4f})')
    return iva_results, best_pair

def clustering_analysis(X, method='kmeans', K_range=range(2, 9)):
    # Submuestreo para silhouette (evita O(N^2) en 60k puntos)
    if len(X) > 5000:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(X), 5000, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    sil_scores = []
    for k in K_range:
        if method == 'kmeans':
            model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        elif method == 'gmm':
            model = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_sub)
        sil_scores.append(silhouette_score(X_sub, labels))
    optimal_k = K_range[np.argmax(sil_scores)]
    return optimal_k, max(sil_scores), sil_scores

def main():
    file_path = 'data/data_situacion_04/Landsat_Cubo_Espaciotemporal_UAO.csv'

    # --- [PUNTO 1] ---
    df, index_data, years = load_and_prepare_data(file_path)

    print('\n[PUNTO 1] CUBO MULTIVARIADO ESPACIO-TEMPORAL')
    print(f'  Dimensiones: {len(df)} pixeles x 44 variables (4 indices x 11 anos)')
    print(f'  Indices: NDVI, NBR, EVI, NDMI (con correccion de escala Landsat Coleccion 2)')

    X_scaled = impute_and_scale(index_data)

    # Combinar todos los indices para PCA/ICA global
    X_combined = np.hstack([X_scaled[idx] for idx in ['evi', 'nbr', 'ndmi', 'ndvi']])

    # --- [PUNTO 2] ---
    print('\n' + '='*60)
    print('[PUNTO 2] IDENTIFICACION DE COMPONENTES LATENTES (ACP/ICA)')
    print('='*60)

    pca, X_pca = apply_pca(X_combined, n_components=5)
    ica, X_ica, ica_curtosis = apply_ica(X_combined, n_components=5)

    # --- [PUNTO 3] ---
    print('\n' + '='*60)
    print('[PUNTO 3] INDEPENDENT VECTOR ANALYSIS (IVA) - MULTIVISTA')
    print('='*60)

    iva_results, best_pair = apply_iva_multivista(index_data, X_scaled, n_comp_iva=5)
    X_iva_best = iva_results[best_pair]['X_iva']

    # --- [PUNTO 4] ---
    print('\n' + '='*60)
    print('[PUNTO 4] DETECCION DE ZONAS DE CAMBIO (CLUSTERING)')
    print('='*60)

    K_range = range(2, 9)

    print('\nEvaluando K optimo...')
    opt_k_kmeans, sil_kmeans, sil_scores_km = clustering_analysis(X_iva_best, 'kmeans', K_range)
    opt_k_gmm, sil_gmm, sil_scores_gmm = clustering_analysis(X_iva_best, 'gmm', K_range)
    print(f'  K-means: K={opt_k_kmeans} (Sil={sil_kmeans:.4f})')
    print(f'  GMM:     K={opt_k_gmm} (Sil={sil_gmm:.4f})')

    optimal_k = 4  # K=4 para interpretabilidad ecologica
    print(f'\nAplicando clustering con K={optimal_k}...')

    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_iva_best)

    gmm = GaussianMixture(n_components=optimal_k, random_state=RANDOM_STATE)
    labels_gmm = gmm.fit_predict(X_iva_best)

    metrics_table = pd.DataFrame({
        'K-means': {
            'silhouette': silhouette_score(X_iva_best, labels_kmeans),
            'davies_bouldin': davies_bouldin_score(X_iva_best, labels_kmeans),
            'calinski_harabasz': calinski_harabasz_score(X_iva_best, labels_kmeans)
        },
        'GMM': {
            'silhouette': silhouette_score(X_iva_best, labels_gmm),
            'davies_bouldin': davies_bouldin_score(X_iva_best, labels_gmm),
            'calinski_harabasz': calinski_harabasz_score(X_iva_best, labels_gmm)
        }
    })
    print(metrics_table.round(4))

    best_method = 'kmeans' if metrics_table.loc['silhouette', 'K-means'] >= metrics_table.loc['silhouette', 'GMM'] else 'gmm'
    df['cluster'] = labels_kmeans if best_method == 'kmeans' else labels_gmm
    print(f'Metodo seleccionado: {best_method.upper()}')

    # --- [PUNTO 5] ---
    print('\n' + '='*60)
    print('[PUNTO 5] EVALUACION CUANTITATIVA DEL CAMBIO')
    print('='*60)

    indices_names = ['ndvi', 'nbr', 'evi', 'ndmi']
    cluster_stats = {}

    for c in range(optimal_k):
        mask = df['cluster'] == c
        count = mask.sum()
        pct = count / len(df) * 100
        print(f'\n--- Cluster {c}: {count} pixeles ({pct:.1f}%) ---')

        cluster_stats[c] = {'count': count, 'pct': pct, 'indices': {}}

        for idx_name in indices_names:
            cols = index_data[idx_name]['cols']
            vals = df[mask][cols].values
            means = np.nanmean(vals, axis=0)
            start_v = means[0]
            end_v = means[-1]
            change_pct = ((end_v - start_v) / abs(start_v)) * 100
            n_years = len(years)
            annual_rate = ((end_v / abs(start_v)) ** (1 / (n_years - 1)) - 1) * 100 if start_v != 0 else 0

            cluster_stats[c]['indices'][idx_name] = {
                'means': means,
                'start': start_v,
                'end': end_v,
                'change_pct': change_pct,
                'annual_rate': annual_rate
            }
            print(f'  {idx_name.upper()}: {start_v:.4f} -> {end_v:.4f} | Cambio={change_pct:+.2f}% | Tasa anual={annual_rate:+.2f}%')

    # --- VISUALIZACIONES ---
    print('\n[Generando visualizaciones...]')

    # 1. Mapa de clusters
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    colors_cluster = plt.cm.tab10(np.linspace(0, 1, optimal_k))
    for c in range(optimal_k):
        mask = df['cluster'] == c
        ax.scatter(df.loc[mask, 'lon'], df.loc[mask, 'lat'],
                  c=[colors_cluster[c]], s=3, alpha=0.5, label=f'C{c}', edgecolors='none')
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.set_title('Mapa Tematico de Clusters - Trayectorias de Cambio Forestal')
    ax.legend(markerscale=5)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}situacion_04_mapa_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Mapa clusters OK')

    # 2. Trayectorias de los 4 indices por cluster
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for ax_idx, idx_name in enumerate(indices_names):
        ax = axes[ax_idx]
        for c in range(optimal_k):
            means = cluster_stats[c]['indices'][idx_name]['means']
            change = cluster_stats[c]['indices'][idx_name]['change_pct']
            ax.plot(years, means, 'o-', color=colors_cluster[c],
                   label=f'C{c} ({change:+.2f}%)', linewidth=2, markersize=6)
        ax.set_xlabel('Ano')
        ax.set_ylabel(idx_name.upper())
        ax.set_title(f'Trayectoria Temporal - {idx_name.upper()}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}situacion_04_trayectorias_indices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Trayectorias OK')

    # 3. Mapas latentes (PC1, PC2, IC1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ['PC1 - Varianza Dominante', 'PC2 - Segundo Modo', 'IC1 - Fuente Independiente']
    data_maps = [X_pca[:, 0], X_pca[:, 1], X_ica[:, 0]]
    for ax_i, data_map, title in zip(axes, data_maps, titles):
        sc = ax_i.scatter(df['lon'], df['lat'], c=data_map, cmap='RdBu_r',
                         s=3, alpha=0.5, edgecolors='none')
        ax_i.set_xlabel('Longitud')
        ax_i.set_ylabel('Latitud')
        ax_i.set_title(title)
        plt.colorbar(sc, ax=ax_i)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}situacion_04_mapas_latentes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Mapas latentes OK')

    # 4. Varianza PCA
    evr = pca.explained_variance_ratio_
    cumsum = np.cumsum(evr)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, 6), evr, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Componente')
    axes[0].set_ylabel('Varianza Explicada')
    axes[0].set_title('Varianza Explicada por Componente')
    axes[1].plot(range(1, 6), cumsum, 'o-', color='darkred', linewidth=2, markersize=8)
    axes[1].axhline(y=0.85, color='gray', linestyle='--', alpha=0.5, label='85%')
    axes[1].set_xlabel('Componentes')
    axes[1].set_ylabel('Varianza Acumulada')
    axes[1].set_title('Varianza Explicada Acumulada')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}situacion_04_varianza_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Varianza PCA OK')

    # 5. Comparacion clustering (K-means vs GMM)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(K_range), sil_scores_km, 'o-', label='K-means', linewidth=2, markersize=8)
    ax.plot(list(K_range), sil_scores_gmm, 's--', label='GMM', linewidth=2, markersize=8)
    ax.axvline(x=opt_k_kmeans, color='blue', linestyle=':', alpha=0.5, label=f'K* K-means={opt_k_kmeans}')
    ax.axvline(x=opt_k_gmm, color='orange', linestyle=':', alpha=0.5, label=f'K* GMM={opt_k_gmm}')
    ax.set_xlabel('K')
    ax.set_ylabel('Silhouette')
    ax.set_title('Comparacion K-means vs GMM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}situacion_04_comparacion_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Comparacion clustering OK')

    # 6. Resumen de cambio por cluster (barplot comparativo)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    changes = {idx: [cluster_stats[c]['indices'][idx]['change_pct'] for c in range(optimal_k)]
               for idx in indices_names}
    annual_rates = {idx: [cluster_stats[c]['indices'][idx]['annual_rate'] for c in range(optimal_k)]
                    for idx in indices_names}

    x = np.arange(optimal_k)
    width = 0.2
    for i, idx in enumerate(indices_names):
        axes[0].bar(x + i*width, changes[idx], width, label=idx.upper())
        axes[1].bar(x + i*width, annual_rates[idx], width, label=idx.upper())
    for ax in axes:
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels([f'C{c}' for c in range(optimal_k)])
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_title('Cambio Total por Indice y Cluster (%)')
    axes[0].set_ylabel('Cambio Total (%)')
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_title('Tasa Anual por Indice y Cluster (%)')
    axes[1].set_ylabel('Tasa Anual (%)')
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}situacion_04_cambio_indices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Cambio indices OK')

    # --- [PUNTO 6] ---
    print('\n' + '='*60)
    print('[PUNTO 6] INTERPRETACION ECOLOGICA Y CONCLUSIONES')
    print('='*60)
    print()

    for c in range(optimal_k):
        ndvi_chg = cluster_stats[c]['indices']['ndvi']['change_pct']
        evi_chg = cluster_stats[c]['indices']['evi']['change_pct']
        nbr_chg = cluster_stats[c]['indices']['nbr']['change_pct']
        ndmi_chg = cluster_stats[c]['indices']['ndmi']['change_pct']

        # Clasificacion basada en consistencia multi-indice
        green_up = sum([ndvi_chg > 2, evi_chg > 2, nbr_chg > 2, ndmi_chg > 2])
        green_down = sum([ndvi_chg < -2, evi_chg < -2, nbr_chg < -2, ndmi_chg < -2])

        if green_up >= 3:
            label = 'REGENERACION ACTIVA'
        elif green_down >= 3:
            label = 'DEGRADACION'
        else:
            label = 'ESTABILIDAD / TRANSICION'

        pct = cluster_stats[c]['pct']
        print(f'Cluster {c} ({pct:.1f}%): {label}')
        print(f'  NDVI: {ndvi_chg:+.2f}% | EVI: {evi_chg:+.2f}% | NBR: {nbr_chg:+.2f}% | NDMI: {ndmi_chg:+.2f}%')

    print(f'\nLimitaciones:')
    print(f'  - Correccion de escala Landsat Coleccion 2 aplicada (factor 0.0000275 - 0.2)')
    print(f'  - Valores faltantes imputados con mediana')
    print(f'  - Resolucion temporal: 1 observacion anual')
    print(f'\nAplicaciones:')
    print(f'  - Monitoreo multi-indice de cobertura forestal')
    print(f'  - Deteccion temprana de deforestacion')
    print(f'  - Evaluacion de efectividad de reforestacion')

    print('\nAnalisis completado exitosamente.')

if __name__ == '__main__':
    main()
