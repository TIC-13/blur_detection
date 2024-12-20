# Este código cria um novo repositório clonado de imagens, mas todas borradas com diferentes níves de desfoque

import os
import random
import cv2
import numpy as np
import pandas as pd

# Desfoque gaussiano
def gaussian_blur(image, kernel_size=(5, 5), sigma=5.0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

# Desfoque de movimento não linear
def motion_blur(image, kernel_size=45):
    # Cria uma matriz de kernel de movimento
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    
    # Aplica o kernel na imagem
    return cv2.filter2D(image, -1, kernel)

# Vamos escolher o tipo de blur aleatoriamente
def apply_random_blur(image_path):
    image = cv2.imread(image_path)
    
    if random.choice([True, False]):
        # Aplica o desfoque gaussiano com parâmetros aleatórios
        kernel_size = random.choice([(3, 3), (5, 5), (7, 7)])
        sigma = random.uniform(5.0, 10.0)
        blurred_image = gaussian_blur(image, kernel_size, sigma)
    else:
        # Aplica o desfoque de movimento não linear com tamanho de kernel aleatório
        kernel_size = random.randint(30, 60)
        blurred_image = motion_blur(image, kernel_size)
    
    return blurred_image

def process_images_from_csv(csv_file, images_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Lê o CSV e obtém os nomes das imagens da coluna 11 (índice 10)
    df = pd.read_csv(csv_file)
    image_names = df.iloc[:, 10]  # Coluna 11 no CSV (índice começa em 0)

    for image_name in image_names:
        # Converte o nome da imagem pra string e remove valores nulos
        if pd.isnull(image_name):
            print(f"Nome da imagem inválido ou nulo. Pulando essa entrada.")
            continue

        image_name = str(image_name).strip()  # Converte pra string e remove espaços em branco

        image_path = os.path.join(images_directory, image_name)
        
        if os.path.exists(image_path):
            blurred_image = apply_random_blur(image_path)
            
            output_path = os.path.join(output_directory, image_name)
            cv2.imwrite(output_path, blurred_image)
            print(f"Imagem {image_name} processada e salva em {output_path}")
        else:
            print(f"Imagem {image_name} não encontrada em {images_directory}")

# Paths
csv_file = "lesions.csv"
images_directory = "/Images"
output_directory = "/synthetically_blurred" # Importante: crie essa pasta antes de rodar o código

process_images_from_csv(csv_file, images_directory, output_directory)
