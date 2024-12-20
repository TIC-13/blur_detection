# Este código calcula muitas métricas de qualidade de imagem (IQA) e também cria repositório dos resultados das imagens FFT, Sobel e Laplacian. Ele foi precursor do algoritmo dos 5 juízes. É bem pesado.

import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import torch
from torchvision import transforms
import pyiqa as iqa

# Salva imagens normalizadas
def save_image(image_array, image_path):
    # Normaliza a imagem para o intervalo 0-255
    normalized_image = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)
    normalized_image = normalized_image.astype(np.uint8)
    cv2.imwrite(image_path, normalized_image)

# Calcula o Laplaciano, retorna a imagem e o valor da variância
def calculate_var_laplacian(image_path):
    image = Image.open(image_path)
    gray_image = np.array(image.convert('L'))

    # Ajusta o tamanho do kernel (ksize)
    ksize = 7

    # Aplica o operador Laplaciano
    laplacian_image = cv2.Laplacian(gray_image, ddepth=cv2.CV_64F, ksize=ksize)

    # Tomar o valor absoluto
    abs_laplacian = np.abs(laplacian_image)

    # Normaliza a imagem
    abs_laplacian = cv2.normalize(abs_laplacian, None, 0, 255, cv2.NORM_MINMAX)
    abs_laplacian = abs_laplacian.astype(np.uint8)

    # Calcula a variância
    laplacian_var = laplacian_image.var()

    return abs_laplacian, laplacian_var

# NIQE e ILNIQE
def calculate_niqe_ilniqe(image_path, niqe_model, ilniqe_model, device):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)
    niqe_score = niqe_model(img_tensor).item()
    ilniqe_score = ilniqe_model(img_tensor).item()
    return niqe_score, ilniqe_score

# BRISQUE
def calculate_brisque(image_path, brisque_model, device):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)
    score = brisque_model(img_tensor).item()
    return score

# FFT, retorna o espectro de magnitude e o valor médio
def calculate_fft(image_path):
    image = Image.open(image_path)
    gray_image = np.array(image.convert('L'))
    fft_image = np.fft.fft2(gray_image)
    fft_shift = np.fft.fftshift(fft_image)
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1e-10)
    high_freq_content = np.mean(magnitude_spectrum)
    return magnitude_spectrum, high_freq_content

# Sobel, retorna a imagem de magnitude do gradiente e o valor médio
def calculate_sobel(image_path):
    image = Image.open(image_path)
    gray_image = np.array(image.convert('L'))

    # Ajusta o tamanho do kernel (ksize)
    ksize = 7

    # Aplica o operador Sobel nas direções x e y
    sobel_x = cv2.Sobel(gray_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    sobel_y = cv2.Sobel(gray_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)

    # Calcula a magnitude do gradiente
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Pega o valor absoluto
    sobel_magnitude = np.abs(sobel_magnitude)

    # Normaliza a imagem
    sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    sobel_magnitude = sobel_magnitude.astype(np.uint8)

    # Calcula o valor médio
    sobel_mean = sobel_magnitude.mean()

    return sobel_magnitude, sobel_mean

def process_images_metrics(image_directory, image_list, brisque_model, niqe_model, ilniqe_model, device):
    results = []

    # Garante que os diretórios de saída existam
    os.makedirs('/descobrindo_thresholds/Laplacian_Blurred_Images', exist_ok=True)
    os.makedirs('/descobrindo_thresholds/FFT_Blurred_Images', exist_ok=True)
    os.makedirs('/descobrindo_thresholds/Sobel_Blurred_Images', exist_ok=True)

    for img_file in image_list:
        image_path = os.path.join(image_directory, img_file)

        if not os.path.isfile(image_path):
            print(f"A imagem {img_file} não foi encontrada no diretório {image_directory}. Pulando.")
            continue

        print(f"Processando métricas da imagem: {img_file}")

        # Calcula as métricas
        brisque_score = calculate_brisque(image_path, brisque_model, device)
        niqe_score, ilniqe_score = calculate_niqe_ilniqe(image_path, niqe_model, ilniqe_model, device)
        laplacian_image, laplacian_var = calculate_var_laplacian(image_path)
        fft_image, fft_mean = calculate_fft(image_path)
        sobel_image, sobel_mean = calculate_sobel(image_path)

        # Salva as imagens resultantes
        base_name = os.path.splitext(img_file)[0]
        laplacian_image_name = f'{base_name}_laplacian.png'
        fft_image_name = f'{base_name}_fft.png'
        sobel_image_name = f'{base_name}_sobel.png'

        laplacian_image_path = os.path.join('/descobrindo_thresholds/Laplacian_Blured_Images', laplacian_image_name)
        fft_image_path = os.path.join('/descobrindo_thresholds/FFT_Blurred_Images', fft_image_name)
        sobel_image_path = os.path.join('/descobrindo_thresholds/Sobel_Blurred_Images', sobel_image_name)

        save_image(laplacian_image, laplacian_image_path)
        save_image(fft_image, fft_image_path)
        save_image(sobel_image, sobel_image_path)

        # Adiciona os resultados ao dicionário
        results.append({
            "img_file": img_file,
            "BRISQUE": brisque_score,
            "NIQE": niqe_score,
            "ILNIQE": ilniqe_score,
            "Laplacian_Value": laplacian_var,
            "FFT_Value": fft_mean,
            "Sobel_Value": sobel_mean,
            "Laplacian_Image": laplacian_image_path,
            "FFT_Image": fft_image_path,
            "Sobel_Image": sobel_image_path
        })

        print(f"Métricas da imagem {img_file} processadas!")

    return results

# Aplica o modelo TFLite às imagens
def apply_model_to_images(image_directory, image_list, interpreter):
    results = []

    # Obtém o shape de entrada do modelo
    input_shape = interpreter.get_input_details()[0]['shape']

    for img_file in image_list:
        image_path = os.path.join(image_directory, img_file)

        if not os.path.isfile(image_path):
            print(f"A imagem {img_file} não foi encontrada no diretório {image_directory}. Pulando.")
            continue

        print(f"Aplicando modelo na imagem: {img_file}")

        # Pré-processamento e aplicação do modelo
        input_data = preprocess_image_for_model(image_path, input_shape)
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
        probabilities = output_data.tolist()

        # Adiciona os resultados ao dicionário
        results.append({
            "img_file": img_file,
            "Model_Probabilities": probabilities
        })

        print(f"Modelo aplicado na imagem {img_file}!")

    return results

# Pré-processamento para o modelo
def preprocess_image_for_model(image_path, input_shape):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_shape[1], input_shape[2]))
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def main():
    # Caminhos dos diretórios e modelos
    images_dir = 'synthetically_blurred' # Aqui, você deve colocar o caminho para o diretório de imagens. Podem ser as imagens borradas, croppadas ou naturais
    csv_file = 'lesions.csv' 
    model_path = 'blur.tflite'
    output_csv = 'descobrindo_thresholds/IQA_blurred_metrics_and_model_cropped.csv' # Mude o caminho para nome correto do CSV

    # Lê os nomes das imagens a partir do arquivo CSV (coluna 11, índice 10)
    df_csv = pd.read_csv(csv_file)
    image_names = df_csv.iloc[:, 10].astype(str).tolist()

    # Remove valores que são 'nan' (resultantes de NaNs)
    image_names = [name for name in image_names if name.lower() != 'nan']

    # Inicializa os modelos de IQA (BRISQUE, NIQE, ILNIQE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    niqe_model = iqa.create_metric('niqe', device=device)
    ilniqe_model = iqa.create_metric('ilniqe', device=device)
    brisque_model = iqa.create_metric('brisque', device=device)

    # Processa as imagens (calcula as métricas e salva as imagens)
    metrics_results = process_images_metrics(images_dir, image_names, brisque_model, niqe_model, ilniqe_model, device)

    # Inicializa o modelo TFLite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Aplica o modelo às imagens
    model_results = apply_model_to_images(images_dir, image_names, interpreter)

    # Combina os resultados das métricas com os resultados do modelo
    # Converte listas de resultados em DataFrames
    df_metrics = pd.DataFrame(metrics_results)
    df_model = pd.DataFrame(model_results)

    # Faz o merge dos DataFrames com base na coluna 'img_file'
    df_final = pd.merge(df_metrics, df_model, on='img_file', how='left')

    # Salva o DataFrame final no CSV
    df_final.to_csv(output_csv, index=False)
    print(f"Resultados salvos em {output_csv}")

if __name__ == "__main__":
    main()