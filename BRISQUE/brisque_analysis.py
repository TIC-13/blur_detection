import pandas as pd
import numpy as np

# Caminho para o arquivo CSV
results = '/seu/caminho/Análise_com_BRISQUE/results_brisque.csv' # Lembrar de atualizar para o seu path

# Lê o CSV
df = pd.read_csv(results)

# Função para calcular média, mediana e desvio padrão
def compute_statistics(data):
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    return mean, median, std_dev

# Mapeia categorias e qualidade
quality_mapping = {
    'Alta': 'Alta qualidade',
    'Boa': 'Boa qualidade',
    'Média': 'Média qualidade',
    'Baixa': 'Baixa qualidade',
    'Negativo': 'Negativo'
}

# Aplica o mapeamento
df['Qualidade'] = df['Categoria'].map(quality_mapping)

# Contagem total de imagens e por qualidade
total_count = len(df)
quality_counts = df['Qualidade'].value_counts()
quality_percentages = (quality_counts / total_count * 100).sort_index()

# Filtra os valores BRISQUE não negativos
df_non_negative = df[df['BRISQUE'] >= 0]

# Cálculo de estatísticas sem valores negativos
total_scores_non_negative = df_non_negative['BRISQUE']
mean_total, median_total, std_dev_total = compute_statistics(total_scores_non_negative)

# Impressão dos resultados
print("Contagem total de imagens por qualidade:")
for quality, count in quality_counts.items():
    percentage = quality_percentages[quality]
    print(f"{quality}: {count} ({percentage:.2f}%)")

print("\nEstatísticas totais (sem valores negativos):")
print(f"Média: {mean_total:.2f}")
print(f"Mediana: {median_total:.2f}")
print(f"Desvio Padrão: {std_dev_total:.2f}")