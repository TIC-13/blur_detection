import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from pyiqa.archs.niqe_arch import NIQE, ILNIQE

input_folder = 'images' # Atenção: atualize o caminho para o diretório com as imagens
output_csv = 'NIQE_E_ILNIQE/resultados_avaliacao_niqe_ilniqe.csv'
threshold_niqe = 5.5  # Exemplo de valor de limiar para NIQE
threshold_ilniqe = 65.0  # Exemplo de valor de limiar para ILNIQE

# Inicialização dos modelos NIQE e ILNIQE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

niqe_model = NIQE().to(device)
ilniqe_model = ILNIQE().to(device)

# Função para calcular a métrica NIQE
def calculate_niqe(img_tensor):
    return niqe_model(img_tensor).item()

# Função para calcular a métrica ILNIQE
def calculate_ilniqe(img_tensor):
    return ilniqe_model(img_tensor).item()

# Função para processar uma imagem
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),  # Ajuste conforme necessário
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    niqe_score = calculate_niqe(img_tensor)
    ilniqe_score = calculate_ilniqe(img_tensor)

    niqe_quality = 'Baixa' if niqe_score > threshold_niqe else 'Alta'
    ilniqe_quality = 'Baixa' if ilniqe_score > threshold_ilniqe else 'Alta'

    return niqe_score, ilniqe_score, niqe_quality, ilniqe_quality

results = []

# Processamento das imagens no diretório
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, file)
            print(f"Processando imagem: {file}")
            niqe_score, ilniqe_score, niqe_quality, ilniqe_quality = process_image(image_path)
            print(f"Imagem: {file}")
            print(f"NIQE: {niqe_score:.2f} - Qualidade: {niqe_quality}")
            print(f"ILNIQE: {ilniqe_score:.2f} - Qualidade: {ilniqe_quality}")
            results.append({
                'Imagem': file,
                'NIQE': niqe_score,
                'ILNIQE': ilniqe_score,
                'Qualidade_NIQE': niqe_quality,
                'Qualidade_ILNIQE': ilniqe_quality
            })

# Salvando os resultados
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

# Calculando estatísticas
niqe_scores = df['NIQE']
ilniqe_scores = df['ILNIQE']

niqe_mean = niqe_scores.mean()
niqe_median = niqe_scores.median()
niqe_std = niqe_scores.std()

ilniqe_mean = ilniqe_scores.mean()
ilniqe_median = ilniqe_scores.median()
ilniqe_std = ilniqe_scores.std()

stats = {
    'Imagem': ['Estatísticas'],
    'NIQE': [f'Média: {niqe_mean:.2f}, Mediana: {niqe_median:.2f}, Desvio Padrão: {niqe_std:.2f}'],
    'ILNIQE': [f'Média: {ilniqe_mean:.2f}, Mediana: {ilniqe_median:.2f}, Desvio Padrão: {ilniqe_std:.2f}'],
    'Qualidade_NIQE': ['-'],
    'Qualidade_ILNIQE': ['-']
}

df_stats = pd.DataFrame(stats)
df_stats.to_csv(output_csv, mode='a', header=False, index=False)

print(f"Resultados e estatísticas salvos em {output_csv}")
