import pandas as pd
import matplotlib.pyplot as plt

# Caminho do CSV contendo os resultados das métricas
csv_file = 'NIQE_E_ILNIQE/resultados_avaliacao_niqe_ilniqe.csv'

# Carrega os resultados do CSV e converter as colunas NIQE e ILNIQE para float
df = pd.read_csv(csv_file)
df['NIQE'] = pd.to_numeric(df['NIQE'], errors='coerce')
df['ILNIQE'] = pd.to_numeric(df['ILNIQE'], errors='coerce')

# Cria histograma para NIQE
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(df['NIQE'].dropna(), bins=[x * 0.5 for x in range(int(df['NIQE'].min() // 0.5), int(df['NIQE'].max() // 0.5) + 2)], edgecolor='black')
plt.title('Distribuição de NIQE')
plt.xlabel('NIQE')
plt.ylabel('Quantidade de Imagens')

# Cria histograma para ILNIQE
plt.subplot(1, 2, 2)
plt.hist(df['ILNIQE'].dropna(), bins=[x * 5 for x in range(int(df['ILNIQE'].min() // 5), int(df['ILNIQE'].max() // 5) + 2)], edgecolor='black')
plt.title('Distribuição de ILNIQE')
plt.xlabel('ILNIQE')
plt.ylabel('Quantidade de Imagens')

# Exibir os gráficos
plt.tight_layout()
plt.show()
