"""
Modulis vizualizacijų kūrimui
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import warnings
warnings.filterwarnings('ignore')

# Nustatome stilių
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_cluster_results(features_df: pd.DataFrame, clustering_results: dict, 
                        output_dir: str):
    """Vizualizuoja klasterizavimo rezultatus"""
    
    # 1. K-means rezultatai - scatter plot
    if 'kmeans' in clustering_results:
        result = clustering_results['kmeans']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('K-Means Klasterizavimo Rezultatai', fontsize=16, fontweight='bold')
        
        # PCA vizualizacija
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df.values)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=result['labels'], 
                                   cmap='viridis', s=100, alpha=0.6)
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[0, 0].set_title('K-means klasteriai (PCA projekcija)')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Klasterių dydžiai
        cluster_counts = pd.Series(result['labels']).value_counts().sort_index()
        axes[0, 1].bar(range(len(cluster_counts)), cluster_counts.values)
        axes[0, 1].set_xlabel('Klasteris')
        axes[0, 1].set_ylabel('Valiutų skaičius')
        axes[0, 1].set_title('Klasterių dydžiai')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Heatmap - charakteristikų vidurkiai pagal klasterius
        cluster_features = pd.DataFrame({
            'currency': features_df.index,
            'cluster': result['labels']
        })
        cluster_means = cluster_features.groupby('cluster')[features_df.index].mean()
        
        # Palengvintai - pasirenkame svarbiausias charakteristikas
        important_features = features_df.columns[:min(10, len(features_df.columns))]
        cluster_means_data = []
        for cluster_id in range(len(result['labels'])):
            cluster_currencies = cluster_features[cluster_features['cluster'] == cluster_id]['currency']
            if len(cluster_currencies) > 0:
                cluster_mean = features_df.loc[cluster_currencies, important_features].mean()
                cluster_means_data.append(cluster_mean)
        
        if cluster_means_data:
            cluster_means_df = pd.DataFrame(cluster_means_data)
            sns.heatmap(cluster_means_df.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                       ax=axes[1, 0], cbar_kws={'label': 'Vidutinė reikšmė'})
            axes[1, 0].set_xlabel('Klasteris')
            axes[1, 0].set_ylabel('Charakteristikos')
            axes[1, 0].set_title('Charakteristikų vidurkiai pagal klasterius')
        
        # Metrikos
        metrics = ['silhouette_score', 'calinski_harabasz_score']
        metric_values = [result.get(m, 0) for m in metrics]
        axes[1, 1].bar(metrics, metric_values, color=['skyblue', 'lightgreen'])
        axes[1, 1].set_ylabel('Reikšmė')
        axes[1, 1].set_title('Vertinimo metrikos')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kmeans_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Hierarchinė klasterizavimas - dendrograma
    if 'hierarchical' in clustering_results:
        result = clustering_results['hierarchical']
        if 'linkage_matrix' in result:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            dendrogram(result['linkage_matrix'], 
                      labels=features_df.index.tolist(),
                      ax=ax,
                      leaf_rotation=90,
                      leaf_font_size=10)
            ax.set_title('Hierarchinio Klasterizavimo Dendrograma', fontsize=14, fontweight='bold')
            ax.set_xlabel('Valiutos poros')
            ax.set_ylabel('Atstumas')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'hierarchical_dendrogram.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. Metodų palyginimas
    methods_data = []
    for method_name, result in clustering_results.items():
        methods_data.append({
            'method': result['method'],
            'silhouette': result.get('silhouette_score', np.nan),
            'calinski': result.get('calinski_harabasz_score', np.nan),
            'davies': result.get('davies_bouldin_score', np.nan)
        })
    
    if methods_data:
        methods_df = pd.DataFrame(methods_data)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Klasterizavimo Metodų Palyginimas', fontsize=16, fontweight='bold')
        
        metrics = ['silhouette', 'calinski', 'davies']
        metric_names = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = methods_df[metric].dropna()
            if len(values) > 0:
                axes[idx].bar(range(len(values)), values.values, color='steelblue')
                axes[idx].set_xticks(range(len(values)))
                axes[idx].set_xticklabels(methods_df.loc[values.index, 'method'], rotation=45, ha='right')
                axes[idx].set_ylabel(name)
                axes[idx].set_title(f'{name} palyginimas')
                axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'methods_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Charakteristikų koreliacijų heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    corr_matrix = features_df.corr()
    
    # Pasirenkame tik svarbiausias charakteristikas (jei per daug)
    if len(corr_matrix) > 20:
        # Raskime didžiausias koreliacijas
        important_features = corr_matrix.abs().sum().nlargest(20).index
        corr_matrix = corr_matrix.loc[important_features, important_features]
    
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
               square=True, linewidths=0.5, cbar_kws={'label': 'Koreliacija'}, ax=ax)
    ax.set_title('Charakteristikų Koreliacijų Matrica', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'features_correlation.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()


def visualize_results(features_df: pd.DataFrame, clustering_results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    plot_cluster_results(features_df, clustering_results, output_dir)

