import torch
import pyiqa as iqa # Lembrar de fazer o pip install pyiqa
import os
# import shutil
import csv

'''
OBS: O código cria um novo diretório com as imagens categorizadas para facilitar a análise da coerência. 
Isso pode não ser interessante por consumir mais armazenamento ao copiá-las. Por isso, as linhas que
fazem a cópia estarão comentadas. Para ativá-las novamente, é só retirar os comentários.
'''

if torch.cuda.is_available():
    device = torch.device("cuda")
# Caso não haja CUDA
else:
    device = torch.device("cpu")

# Cria a métrica BRISQUE
brisque = iqa.create_metric('brisque', device=device)

# Caminho para o CSV com as imagens válidas
valid_images_csv = '/seu/caminho/para/lesions.csv' # Lembrar de atualizar o caminho para o csv 

# Diretório das imagens
images_dir = '/seu/caminho/para/images' # Lembrar de atualizar o caminho para o diretório das imagens

# Diretório de saída
output_dir = '/seu/diretório/do/output/Análise_com_BRISQUE'

# Caminho do arquivo de resultados
results = os.path.join(output_dir, 'results_brisque.csv')

# Cria diretórios de saída, se não existirem
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Função para determinar a categoria de qualidade
def quality_category(score):
    if score < 0:
        return "Negativo", "Negativo"
    elif score <= 20:
        return "Alta", "Alta qualidade"
    elif 20 < score <= 40:
        return "Boa", "Boa qualidade"
    elif 40 < score <= 60:
        return "Média", "Média qualidade"
    else:
        return "Baixa", "Baixa qualidade"

# Lendo o CSV com as imagens válidas (somente a coluna 11)
with open(valid_images_csv, 'r') as csvfile:
    reader = csv.reader(csvfile)
    valid_images = {row[10] for row in reader if len(row) > 10}
 
# Função para processar imagens no diretório
def process_images(directory, output_dir):
    with open(results, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for filename in os.listdir(directory):
            if filename in valid_images:
                image_path = os.path.join(directory, filename)
                score = brisque(image_path).item()

                # Determina a categoria de qualidade
                category, folder = quality_category(score)
                #category_dir = os.path.join(output_dir, folder)
                #os.makedirs(category_dir, exist_ok=True)

                # Copia a imagem para o diretório correspondente
                #shutil.copy(image_path, category_dir)

                # Escreve os resultados no CSV
                csvwriter.writerow([filename, score, category])

                # Printando a imagem, o score BRISQUE e a categoria
                print(f"Imagem: {filename}, BRISQUE: {score}, Categoria: {category}")

# Inicializa o CSV com cabeçalho
with open(results, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Imagem', 'BRISQUE', 'Categoria'])

# Processa as imagens no diretório especificado
process_images(images_dir, output_dir)

print("Processamento concluído e resultados salvos em results_brisque.csv.")
