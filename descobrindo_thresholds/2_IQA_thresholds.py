# Este código compara os resultados de dois csvs e define os melhores thresholds para cada métrica, além de plotar os histogramas e clusters K-Means.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.cluster import KMeans
import os

# Carrega os dados
def load_data(natural_csv, blurred_csv):
    df_natural = pd.read_csv(natural_csv)
    df_blurred = pd.read_csv(blurred_csv)
    return df_natural, df_blurred

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
# Aqui, vão os dois arquivos CSV que você salvou pra comparar as métricas
df_natural, df_blurred = load_data(
    'IQA_metrics_and_model_results_cropped.csv', 
    'IQA_blurred_metrics_and_model_results_cropped.csv'
)

# Define as métricas
metrics = ['BRISQUE', 'NIQE', 'ILNIQE', 'Laplacian_Value', 'FFT_Value', 'Sobel_Value']
natural_stats = calculate_statistics(df_natural, metrics)
blurred_stats = calculate_statistics(df_blurred, metrics)

# Salva os resultados
output_dir = 'descobrindo_thresholds'
save_statistics(natural_stats, os.path.join(output_dir, 'natural_metricas.txt'))
save_statistics(blurred_stats, os.path.join(output_dir, 'blurred_metricas.txt'))

# Plota os histogramas
for metric in metrics:
    plt.figure(figsize=(10, 6))
    
    sns.histplot(df_natural[metric], color='blue', kde=False, label='Natural', stat='count')  # Histogramas por contagem de imagens
    sns.histplot(df_blurred[metric], color='red', kde=False, label='Blurred', stat='count')
    
    plt.title(f'Histograma de {metric} (Contagem)')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, f'histograma_{metric}_contagem.png'))
    plt.close()

# Youden's J Index
def find_best_thresholds(df_natural, df_blurred, metric):
    true_labels = np.concatenate((np.zeros(len(df_natural)), np.ones(len(df_blurred))))
    metric_values = np.concatenate((df_natural[metric], df_blurred[metric]))
    
    fpr, tpr, thresholds = roc_curve(true_labels, metric_values)
    J = tpr - fpr
    idx = np.argmax(J)
    best_threshold = thresholds[idx]
    best_j = J[idx]
    
    return best_threshold, best_j

# Calcula os melhores thresholds
best_thresholds = {}
for metric in metrics:
    best_threshold, best_j = find_best_thresholds(df_natural, df_blurred, metric)
    best_thresholds[metric] = (best_threshold, best_j)
    print(f"Melhor threshold para {metric} (Youden's J): {best_threshold} com J = {best_j}")

# K-Means clustering
for metric in metrics:
    combined_data = pd.concat([df_natural[[metric]], df_blurred[[metric]]])
    combined_data = combined_data.reset_index(drop=True)

    kmeans = KMeans(n_clusters=2, random_state=42)
    combined_data['cluster'] = kmeans.fit_predict(combined_data[[metric]])
    combined_data['label'] = combined_data['cluster'].map({0: 'Natural', 1: 'Blurred'})
    color_map = {'Natural': 'blue', 'Blurred': 'red'}
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=combined_data, x=metric, y=np.zeros(len(combined_data)), hue='label', palette=color_map, s=100)
    plt.title(f'K-Means Clustering de {metric}')
    plt.xlabel(metric)
    plt.yticks([])
    plt.xlim(combined_data[metric].min() - 1, combined_data[metric].max() + 1)
    plt.savefig(os.path.join(output_dir, f'kmeans_clustering_{metric}.png'))
    plt.close()
    
    print(f"Centro do K-Means cluster de {metric}: {kmeans.cluster_centers_}")
