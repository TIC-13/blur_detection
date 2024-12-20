# Este código visa aplicar o algoritmo dos 3 juízes para validar os labels realizados manualmente das imagens anotadas e fornece os resumos das métricas de Acurácia, Precisão, Recall, F1-Score, quantidade de imagens classificadas como "sharp" e "blurred" de acordo com o resultado final.

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import torch
from torchvision import transforms
import pyiqa as iqa
import time
import pandas as pd

# Aplica o modelo e verifica a confiança dos resultados
def apply_model_and_check_confidence(image_path, interpreter):
    input_shape = interpreter.get_input_details()[0]['shape']
    input_data = preprocess_image_for_model(image_path, input_shape)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
    probabilities = output_data.tolist()
    
    prediction = classify_model_probabilities(probabilities)
    confidence = max(probabilities)
    
    return prediction, confidence

# Classifica a imagem com base nas probabilidades do modelo
def classify_model_probabilities(probabilities):
    return "blurred" if probabilities[0] > probabilities[1] else "sharp"

# Pré-processamento pro modelo
def preprocess_image_for_model(image_path, input_shape):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_shape[1], input_shape[2]))
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# ILNIQE
def calculate_ilniqe(image_path, ilniqe_model, device):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]) # Transforma a imagem pra 256x256
    img_tensor = transform(image).unsqueeze(0).to(device) # Adiciona a dimensão do batch
    return ilniqe_model(img_tensor).item()

# FFT
def calculate_fft(image_path):
    image = Image.open(image_path).convert('L') # Converte a imagem pra escala de cinza
    gray_image = np.array(image)
    fft_image = np.fft.fft2(gray_image) # Calcula a FFT
    fft_shift = np.fft.fftshift(fft_image) # Shifta a FFT pro centro da imagem para visualização mais fácil
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1e-10) # Calcula o log da magnitude do espectro da FFT. Adiciona 1e-10 para evitar log(0) e multiplica por 20 para melhor visualização
    return np.mean(magnitude_spectrum)

# Funções para classificar as métricas
def classify_ilniqe(ilniqe_score):
    return "low quality" if ilniqe_score > 91.0 else "good quality"

def classify_fft(fft_score):
    return "low quality" if fft_score < 134.35 else "good quality"

# Função principal para receber uma imagem e aplicar o modelo e métricas
def evaluate_image(image_path, interpreter, ilniqe_model, device):
    # Aplica o modelo
    prediction, confidence = apply_model_and_check_confidence(image_path, interpreter)
    
    # Se a confiança for alta (>= 0.90), usa o resultado do modelo
    if confidence >= 0.90:
        return prediction, confidence, None, None
    
    # Se a confiança for baixa, aplica apenas as métricas ILNIQE e FFT
    ilniqe_score = calculate_ilniqe(image_path, ilniqe_model, device)
    fft_score = calculate_fft(image_path)
    
    # Classifica cada métrica
    ilniqe_quality = classify_ilniqe(ilniqe_score)
    fft_quality = classify_fft(fft_score)
    
    # Conta os votos das métricas e do modelo
    votes = {
        "model": "good quality" if prediction == "sharp" else "low quality",
        "ilniqe": ilniqe_quality,
        "fft": fft_quality
    }
    
    # Decide a classificação final com base na maioria dos votos
    final_classification = "sharp" if list(votes.values()).count("good quality") > 1 else "blurred"
    
    return final_classification, confidence, ilniqe_score, fft_score

# Função principal de execução
def main():
    # Caminho para o arquivo CSV e diretório das imagens
    csv_path = 'imagens_anotadas/imagens_anotadas.csv'
    image_directory = 'images' # Atenção: aqui, coloque o caminho para o diretório com as imagens
    output_csv_path = 'imagens_anotadas/testes/resultados_3_juizes_imagens_anotadas.csv'
    output_txt_path = 'imagens_anotadas/testes/resultados_3_juizes_imagens_anotadas.txt'
    
    # Carrega o arquivo CSV
    df = pd.read_csv(csv_path)
    
    # Inicializa o modelo ILNIQE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ilniqe_model = iqa.create_metric('ilniqe', device=device)
    
    # Inicializa o modelo TFLite
    interpreter = tf.lite.Interpreter(model_path='blur.tflite') # Aqui, coloque o caminho pro modelo tflite
    interpreter.allocate_tensors()
    
    # Mede o tempo de execução total
    start_time = time.time()
    
    # Lista para armazenar os resultados
    results = []
    
    # Arquivo TXT para salvar os resultados detalhados
    with open(output_txt_path, 'w') as txt_file:
        # Loop para processar cada imagem do CSV
        for idx, row in df.iterrows():
            image_name = row['IMG_FILE']
            real_blurring = row['BLURRING']
            image_path = os.path.join(image_directory, image_name)
            
            if not os.path.exists(image_path):
                print(f"Imagem não encontrada: {image_path}")
                txt_file.write(f"Imagem não encontrada: {image_path}\n")
                continue
            
            # Avalia a imagem
            final_result, confidence, ilniqe_score, fft_score = evaluate_image(image_path, interpreter, ilniqe_model, device)
            
            # Armazena o resultado
            results.append([image_name, real_blurring, final_result])
            
            # Exibe o resultado final e as métricas
            print(f"Imagem: {image_name}")
            print(f"Confiança do modelo: {confidence:.2f}")
            txt_file.write(f"Imagem: {image_name}\n")
            txt_file.write(f"Confiança do modelo: {confidence:.2f}\n")
            if ilniqe_score is not None:
                print(f"ILNIQE Score: {ilniqe_score:.2f}")
                txt_file.write(f"ILNIQE Score: {ilniqe_score:.2f}\n")
            if fft_score is not None:
                print(f"FFT Score: {fft_score:.2f}")
                txt_file.write(f"FFT Score: {fft_score:.2f}\n")
            print(f"A imagem foi classificada como: {final_result}")
            txt_file.write(f"A imagem foi classificada como: {final_result}\n")
            print("-" * 50)
            txt_file.write("-" * 50 + "\n")
    
    # Salva os resultados em um novo CSV
    results_df = pd.DataFrame(results, columns=['IMG_FILE', 'BLURRING', 'FINAL_RESULT'])
    results_df.to_csv(output_csv_path, index=False)
    
    # Calcula o tempo total
    execution_time = time.time() - start_time
    print(f"Tempo de execução total: {execution_time} segundos")
    with open(output_txt_path, 'a') as txt_file:
        txt_file.write(f"Tempo de execução total: {execution_time} segundos\n")

if __name__ == "__main__":
    main()