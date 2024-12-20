# Este código tenta comparar os resultados de três csvs e define os melhores thresholds para cada métrica, além de plotar os histogramas e clusters K-Means.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os

# Carrega os dados
def load_data(natural_csv, cropped_csv, blurred_csv):
    df_natural = pd.read_csv(natural_csv)
    df_cropped = pd.read_csv(cropped_csv)
    df_blurred = pd.read_csv(blurred_csv)
    return df_natural, df_cropped, df_blurred

# Estatísticas
def calculate_statistics(df, metrics):
    stats = {}
    for metric in metrics:
        stats[metric] = {
            'média': df[metric].mean(),
            'mediana': df[metric].median(),
            'desvio padrão': df[metric].std(),
            'variância': df[metric].var(),
            'mín': df[metric].min(),
            'max': df[metric].max()
        }
    return stats

# Salva as estatísticas
def save_statistics(stats, filename):
    with open(filename, 'w') as f:
        for metric, values in stats.items():
            f.write(f"{metric}:\n")
            for stat_name, stat_value in values.items():
                f.write(f"  {stat_name}: {stat_value}\n")
            f.write("\n")

# Carrega os dados
df_natural, df_cropped, df_blurred = load_data(
    'descobrindo_thresholds/IQA_metrics_and_model_results_full.csv',
    'descobrindo_thresholds/IQA_metrics_and_model_results_cropped.csv', 
    'descobrindo_thresholds/IQA_blurred_metrics_and_model_results_cropped.csv'
)

# Define as métricas
metrics = ['BRISQUE', 'NIQE', 'ILNIQE', 'Laplacian_Value', 'FFT_Value', 'Sobel_Value']
natural_stats = calculate_statistics(df_natural, metrics)
cropped_stats = calculate_statistics(df_cropped, metrics)
blurred_stats = calculate_statistics(df_blurred, metrics)

# Salva os resultados
output_dir = 'descobrindo_thresholds/graficos_e_resultados'
save_statistics(natural_stats, os.path.join(output_dir, 'natural_metricas_3.txt'))
save_statistics(cropped_stats, os.path.join(output_dir, 'cropped_metricas_3.txt'))
save_statistics(blurred_stats, os.path.join(output_dir, 'blurred_metricas_3.txt'))

# Plota os histogramas
for metric in metrics:
    plt.figure(figsize=(10, 6))
    
    sns.histplot(df_natural[metric], color='blue', kde=False, label='Natural', stat='count')
    sns.histplot(df_cropped[metric], color='green', kde=False, label='Cropped', stat='count')
    sns.histplot(df_blurred[metric], color='red', kde=False, label='Blurred', stat='count')
    
    plt.title(f'Histograma de {metric} (Contagem)')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, f'3_histograma_{metric}_contagem.png'))
    plt.close()

# K-Means clustering
dfs = [df_natural, df_cropped, df_blurred]  # Lista com os DataFrames
for metric in metrics:
    combined_data = pd.concat(dfs)
    combined_data = combined_data.reset_index(drop=True)

    kmeans = KMeans(n_clusters=3, random_state=42)  # Agora 3 clusters
    combined_data['cluster'] = kmeans.fit_predict(combined_data[[metric]])
    combined_data['label'] = combined_data['cluster'].map({0: 'Full', 1: 'Cropped', 2: 'Blurred'})
    color_map = {'Full': 'blue', 'Cropped': 'green', 'Blurred': 'red'}
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=combined_data, x=metric, y=np.zeros(len(combined_data)), hue='label', palette=color_map, s=100)
    plt.title(f'K-Means Clustering de {metric}')
    plt.xlabel(metric)
    plt.yticks([])
    plt.xlim(combined_data[metric].min() - 1, combined_data[metric].max() + 1)
    plt.savefig(os.path.join(output_dir, f'3_kmeans_clustering_{metric}.png'))
    plt.close()
    
    print(f"Centro do K-Means cluster de {metric}: {kmeans.cluster_centers_}")
