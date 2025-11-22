import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from clustering import ClusteringPipeline
from visualization import visualize_results

def prepare_features_from_csv(csv_file: str, output_dir: str = 'csv_clustering_results'):
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        return None, None
    
    timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
    if timestamp_cols:
        for ts_col in timestamp_cols:
            try:
                df[ts_col] = pd.to_datetime(df[ts_col])
            except:
                pass
    
    from features import extract_all_features, calculate_global_features, calculate_session_features
    from config import SESSION_TIMES
    
    if 'symbol' in df.columns:
        features_dict = {}
        all_returns = {}
        benchmark_returns = None
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            if 'best_ask_twvwap' in symbol_data.columns and 'best_bid_twvwap' in symbol_data.columns:
                symbol_data['mid_price'] = (symbol_data['best_ask_twvwap'] + symbol_data['best_bid_twvwap']) / 2
                symbol_data['close'] = symbol_data['mid_price']
                returns = symbol_data['close'].pct_change()
                all_returns[symbol] = returns
                
                if 'BTC' in symbol.upper() and benchmark_returns is None:
                    benchmark_returns = returns
        
        if benchmark_returns is None and all_returns:
            first_currency = list(all_returns.keys())[0]
            benchmark_returns = all_returns[first_currency]
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            if 'best_ask_twvwap' in symbol_data.columns and 'best_bid_twvwap' in symbol_data.columns:
                symbol_data['mid_price'] = (symbol_data['best_ask_twvwap'] + symbol_data['best_bid_twvwap']) / 2
                symbol_data['close'] = symbol_data['mid_price']
                symbol_data['high'] = symbol_data['best_ask_twvwap']
                symbol_data['low'] = symbol_data['best_bid_twvwap']
                symbol_data['open'] = symbol_data['mid_price'].shift(1).fillna(symbol_data['mid_price'].iloc[0])
                
                if 'best_ask_twvwap' in symbol_data.columns and 'best_bid_twvwap' in symbol_data.columns:
                    symbol_data['volume'] = (symbol_data['best_ask_twvwap'] - symbol_data['best_bid_twvwap']).abs()
                else:
                    symbol_data['volume'] = 0
            
            try:
                features = extract_all_features(
                    symbol_data,
                    benchmark_returns=benchmark_returns,
                    all_returns=all_returns,
                    session_times=SESSION_TIMES
                )
                features_dict[symbol] = features
            except Exception as e:
                features_dict[symbol] = {}
        
        features_df = pd.DataFrame.from_dict(features_dict, orient='index')
        
    else:
        if 'best_ask_twvwap' in df.columns and 'best_bid_twvwap' in df.columns:
            df['mid_price'] = (df['best_ask_twvwap'] + df['best_bid_twvwap']) / 2
            df['close'] = df['mid_price']
            df['high'] = df['best_ask_twvwap']
            df['low'] = df['best_bid_twvwap']
            df['open'] = df['mid_price'].shift(1).fillna(df['mid_price'].iloc[0])
            df['volume'] = (df['best_ask_twvwap'] - df['best_bid_twvwap']).abs()
        
        features = extract_all_features(df, session_times=SESSION_TIMES)
        features_df = pd.DataFrame([features])
        features_df.index = ['all_data']
    
    features_df = features_df.fillna(0)
    
    features_path = os.path.join(output_dir, 'features_from_csv.csv')
    features_df.to_csv(features_path)
    
    return features_df, df


def cluster_csv_data(csv_file: str = 'parquet_analysis/data_export.csv', 
                     output_dir: str = 'csv_clustering_results',
                     n_clusters: int = 4):
    if not os.path.exists(csv_file):
        return
    
    features_df, original_df = prepare_features_from_csv(csv_file, output_dir)
    
    if features_df is None or features_df.empty:
        return
    
    from config import CLUSTERING_CONFIG
    
    clustering = ClusteringPipeline(n_clusters=n_clusters, random_state=42)
    clustering_results = clustering.run_all_methods(features_df, CLUSTERING_CONFIG)
    
    results_data = []
    
    for method_name, result in clustering_results.items():
        labels = result['labels']
        currency_names = features_df.index.tolist()
        
        for i, (currency, label) in enumerate(zip(currency_names, labels)):
            results_data.append({
                'currency': currency,
                'cluster': int(label),
                'method': result['method'],
                'silhouette_score': result.get('silhouette_score', np.nan),
                'calinski_harabasz_score': result.get('calinski_harabasz_score', np.nan),
                'davies_bouldin_score': result.get('davies_bouldin_score', np.nan)
            })
    
    results_df = pd.DataFrame(results_data)
    results_path = os.path.join(output_dir, 'clustering_results.csv')
    results_df.to_csv(results_path, index=False)
    
    methods_comparison = []
    for method_name, result in clustering_results.items():
        methods_comparison.append({
            'method': result['method'],
            'silhouette_score': result.get('silhouette_score', np.nan),
            'calinski_harabasz_score': result.get('calinski_harabasz_score', np.nan),
            'davies_bouldin_score': result.get('davies_bouldin_score', np.nan),
            'n_clusters': len(set(result['labels']))
        })
    
    comparison_df = pd.DataFrame(methods_comparison)
    comparison_path = os.path.join(output_dir, 'methods_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
    try:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        visualize_results(features_df, clustering_results, vis_dir)
    except Exception as e:
        pass
    
    print("Skaiciavimai baigti")


def main():
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = 'parquet_analysis/data_export.csv'
    
    n_clusters = 4
    if len(sys.argv) > 2:
        try:
            n_clusters = int(sys.argv[2])
        except:
            pass
    
    if not os.path.exists(csv_file):
        return
    
    cluster_csv_data(csv_file, n_clusters=n_clusters)


if __name__ == "__main__":
    main()
