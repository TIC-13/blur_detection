# Importante: Este código aplica o algoritmo a apenas uma imagem. O código geral_3_juizes.py aplica o algoritmo a um diretório de imagens. Dentro de imagens_anotadas/codigos_juizes, temos códigos que analisam os resultados do algoritmo num conjunto de imagens anotadas.

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import torch
from torchvision import transforms
import pyiqa as iqa
import time

# Aplica o modelo e verifica a confiança
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

# Classificação do modelo com base nas probabilidades
def classify_model_probabilities(probabilities):
    return "blurred" if probabilities[0] > probabilities[1] else "sharp"

# Pré-processa a imagem para o modelo
def preprocess_image_for_model(image_path, input_shape):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_shape[1], input_shape[2]))
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# ILNIQE
def calculate_ilniqe(image_path, ilniqe_model, device):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)
    return ilniqe_model(img_tensor).item()

# FFT
def calculate_fft(image_path):
    image = Image.open(image_path).convert('L') # Converte a imagem pra escala de cinza
    gray_image = np.array(image)
    fft_image = np.fft.fft2(gray_image) # Calcula a FFT
    fft_shift = np.fft.fftshift(fft_image) # Shifta a FFT pro centro da imagem para visualização mais fácil
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1e-10) # Calcula o log da magnitude do espectro da FFT. Adiciona 1e-10 para evitar log(0) e multiplica por 20 para melhor visualização
    return np.mean(magnitude_spectrum)

# Classifica as métricas
def classify_ilniqe(ilniqe_score):
    return "low quality" if ilniqe_score > 91.0 else "good quality"

def classify_fft(fft_score):
    return "low quality" if fft_score < 134.35 else "good quality"

def evaluate_image(image_path, interpreter, ilniqe_model, device):
    # Aplica o modelo
    prediction, confidence = apply_model_and_check_confidence(image_path, interpreter)
    
    # Se a confiança for alta (>= 0.9), usa o resultado do modelo
    if confidence >= 0.9: # Caso algumas imagens não estejam sendo classificadas corretamente, aumentar o valor de confiança (sugestão = 0.98). 0.9 é o valor mais otimizado de classificação.
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

def main():
    # Caminho da imagem a ser avaliada
    image_path = 'images/0a0a8499-c966-4a1d-904b-85e230705071.png' # Atenção: Alterar o caminho da imagem para o caminho da imagem a ser avaliada
    
    # Inicializa o modelo ILNIQE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ilniqe_model = iqa.create_metric('ilniqe', device=device)
    
    # Inicializa o modelo TFLite
    interpreter = tf.lite.Interpreter(model_path='blur.tflite')
    interpreter.allocate_tensors()
    
    # Mede o tempo de execução
    start_time = time.time()

    # Avalia a imagem
    final_result, confidence, ilniqe_score, fft_score = evaluate_image(image_path, interpreter, ilniqe_model, device)
    prediction = apply_model_and_check_confidence(image_path, interpreter)

    # Calcula o tempo total
    execution_time = time.time() - start_time

    # Exibe o resultado final e as métricas
    print(f"Palpite inicial: {prediction}") 
    print(f"Confiança do modelo: {confidence:.2f}")
    if ilniqe_score is not None and fft_score is not None:
        print(f"ILNIQE Score: {ilniqe_score:.2f}")
        print(f"FFT Score: {fft_score:.2f}")
    print(f"A imagem foi classificada como: {final_result}")

    # Exibe o tempo de execução
    print(f"Tempo de execução: {execution_time} segundos")

if __name__ == "__main__":
    main()
