import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

try:
    from tslearn.clustering import TimeSeriesKMeans
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False


class ClusteringPipeline:
    
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.results = {}
    
    def prepare_data(self, features_df: pd.DataFrame):
        features_df = features_df.fillna(0)
        X = self.scaler.fit_transform(features_df.values)
        return X, features_df.index.tolist()
    
    def kmeans_clustering(self, features_df: pd.DataFrame, n_clusters: int = None):
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        X, labels = self.prepare_data(features_df)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        silhouette = silhouette_score(X, cluster_labels)
        calinski = calinski_harabasz_score(X, cluster_labels)
        davies = davies_bouldin_score(X, cluster_labels)
        
        return {
            'labels': cluster_labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies,
            'method': 'K-Means'
        }
    
    def hierarchical_clustering(self, features_df: pd.DataFrame, n_clusters: int = None, 
                               linkage_method: str = 'ward'):
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        X, labels = self.prepare_data(features_df)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        cluster_labels = clustering.fit_predict(X)
        linkage_matrix = linkage(X, method=linkage_method)
        silhouette = silhouette_score(X, cluster_labels)
        calinski = calinski_harabasz_score(X, cluster_labels)
        davies = davies_bouldin_score(X, cluster_labels)
        
        return {
            'labels': cluster_labels,
            'linkage_matrix': linkage_matrix,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies,
            'method': 'Hierarchical'
        }
    
    def dbscan_clustering(self, features_df: pd.DataFrame, eps: float = 0.5, 
                         min_samples: int = 2):
        X, labels = self.prepare_data(features_df)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        if n_clusters > 1:
            silhouette = silhouette_score(X, cluster_labels)
            calinski = calinski_harabasz_score(X, cluster_labels)
            davies = davies_bouldin_score(X, cluster_labels)
        else:
            silhouette = -1
            calinski = 0
            davies = float('inf')
        
        return {
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': list(cluster_labels).count(-1),
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies,
            'method': 'DBSCAN'
        }
    
    def gmm_clustering(self, features_df: pd.DataFrame, n_components: int = None):
        if n_components is None:
            n_components = self.n_clusters
        
        X, labels = self.prepare_data(features_df)
        gmm = GaussianMixture(n_components=n_components, random_state=self.random_state)
        cluster_labels = gmm.fit_predict(X)
        silhouette = silhouette_score(X, cluster_labels)
        calinski = calinski_harabasz_score(X, cluster_labels)
        davies = davies_bouldin_score(X, cluster_labels)
        
        return {
            'labels': cluster_labels,
            'means': gmm.means_,
            'covariances': gmm.covariances_,
            'weights': gmm.weights_,
            'aic': gmm.aic(X),
            'bic': gmm.bic(X),
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies,
            'method': 'GMM'
        }
    
    def pca_kmeans_clustering(self, features_df: pd.DataFrame, n_components: int = None, 
                             n_clusters: int = None):
        if n_clusters is None:
            n_clusters = self.n_clusters
        if n_components is None:
            n_components = min(len(features_df.columns), len(features_df) - 1, 10)
        
        X, labels = self.prepare_data(features_df)
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)
        silhouette = silhouette_score(X_pca, cluster_labels)
        calinski = calinski_harabasz_score(X_pca, cluster_labels)
        davies = davies_bouldin_score(X_pca, cluster_labels)
        
        return {
            'labels': cluster_labels,
            'pca_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies,
            'method': 'PCA + K-Means'
        }
    
    def correlation_network_clustering(self, features_df: pd.DataFrame, 
                                     correlation_threshold: float = 0.7):
        corr_matrix = features_df.corr().abs()
        adjacency = (corr_matrix > correlation_threshold).astype(int)
        from sklearn.cluster import AgglomerativeClustering
        distance_matrix = 1 - corr_matrix.abs()
        distance_matrix = distance_matrix.fillna(0)
        
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage='average',
            metric='precomputed'
        )
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        return {
            'labels': cluster_labels,
            'correlation_matrix': corr_matrix,
            'adjacency_matrix': adjacency,
            'method': 'Correlation Network'
        }
    
    def run_all_methods(self, features_df: pd.DataFrame, config: dict = None):
        if config is None:
            from config import CLUSTERING_CONFIG
            config = CLUSTERING_CONFIG
        
        results = {}
        results['kmeans'] = self.kmeans_clustering(features_df, config['k_means']['n_clusters'])
        results['hierarchical'] = self.hierarchical_clustering(
            features_df, config['k_means']['n_clusters']
        )
        results['dbscan'] = self.dbscan_clustering(
            features_df, 
            eps=config['dbscan']['eps'],
            min_samples=config['dbscan']['min_samples']
        )
        results['gmm'] = self.gmm_clustering(
            features_df, config['gmm']['n_components']
        )
        results['pca_kmeans'] = self.pca_kmeans_clustering(
            features_df, n_clusters=config['k_means']['n_clusters']
        )
        results['correlation_network'] = self.correlation_network_clustering(features_df)
        
        self.results = results
        return results
