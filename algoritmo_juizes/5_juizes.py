import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import torch
from torchvision import transforms
import pyiqa as iqa
import time  # Importa a biblioteca time

# Aplica o modelo e verificar a confiança
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

# Métricas de qualidade de imagem
def calculate_brisque(image_path, brisque_model, device):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)
    return brisque_model(img_tensor).item()

def calculate_niqe_ilniqe(image_path, niqe_model, ilniqe_model, device):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)
    niqe_score = niqe_model(img_tensor).item()
    ilniqe_score = ilniqe_model(img_tensor).item()
    return niqe_score, ilniqe_score

def calculate_fft(image_path):
    image = Image.open(image_path).convert('L')
    gray_image = np.array(image)
    fft_image = np.fft.fft2(gray_image)
    fft_shift = np.fft.fftshift(fft_image)
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1e-10)
    return np.mean(magnitude_spectrum)

# Funções para classificar as métricas
def classify_brisque(brisque_score):
    return "low quality" if brisque_score > 40.7 else "good quality"

def classify_niqe(niqe_score):
    return "low quality" if niqe_score > 7.1 else "good quality"

def classify_ilniqe(ilniqe_score):
    return "low quality" if ilniqe_score > 91.0 else "good quality"

def classify_fft(fft_score):
    return "low quality" if fft_score < 134.35 else "good quality"

# Função principal para receber uma imagem e aplicar o modelo e métricas
def evaluate_image(image_path, interpreter, brisque_model, niqe_model, ilniqe_model, device):
    # Aplica o modelo
    prediction, confidence = apply_model_and_check_confidence(image_path, interpreter)
    
    # Se a confiança for alta (>= 0.9), usa o resultado do modelo
    if confidence >= 0.9:
        return prediction, confidence, None, None, None, None
    
    # Se a confiança for baixa, aplica as métricas
    brisque_score = calculate_brisque(image_path, brisque_model, device)
    niqe_score, ilniqe_score = calculate_niqe_ilniqe(image_path, niqe_model, ilniqe_model, device)
    fft_score = calculate_fft(image_path)
    
    # Classifica cada métrica
    brisque_quality = classify_brisque(brisque_score)
    niqe_quality = classify_niqe(niqe_score)
    ilniqe_quality = classify_ilniqe(ilniqe_score)
    fft_quality = classify_fft(fft_score)
    
    # Conta os votos das métricas e do modelo
    votes = {
        "model": "good quality" if prediction == "sharp" else "low quality",
        "brisque": brisque_quality,
        "niqe": niqe_quality,
        "ilniqe": ilniqe_quality,
        "fft": fft_quality
    }
    
    # Decide a classificação final com base na maioria dos votos
    good_quality_count = list(votes.values()).count("good quality")
    low_quality_count = list(votes.values()).count("low quality")
    
    # A maioria dos votos determina a classificação
    final_classification = "sharp" if good_quality_count > low_quality_count else "blurred"
    
    return final_classification, confidence, brisque_score, niqe_score, ilniqe_score, fft_score

def main():
    # Caminho da imagem a ser avaliada
    image_path = '/images/e4b1a81a-0d5f-4cc3-8e31-e3146b01a13f.png' # Atenção: aqui, coloque o caminho para a imagem a ser avaliada
    
    # Inicializa os modelos de IQA (BRISQUE, NIQE, ILNIQE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    niqe_model = iqa.create_metric('niqe', device=device)
    ilniqe_model = iqa.create_metric('ilniqe', device=device)
    brisque_model = iqa.create_metric('brisque', device=device)
    
    # Inicializa o modelo TFLite
    interpreter = tf.lite.Interpreter(model_path='blur.tflite') # Atenção: aqui, coloque o caminho pro modelo tflite
    interpreter.allocate_tensors()
    
    # Mede o tempo de execução
    start_time = time.time()

    # Avalia a imagem
    final_result, confidence, brisque_score, niqe_score, ilniqe_score, fft_score = evaluate_image(image_path, interpreter, brisque_model, niqe_model, ilniqe_model, device)
    prediction = apply_model_and_check_confidence(image_path, interpreter)

    # Calcula o tempo total
    execution_time = time.time() - start_time

    # Exibe o resultado final e as métricas
    print(f"Palpite inicial e Confiança do modelo: {prediction}")
    print(f"Confiança do modelo: {confidence:.2f}")
    if brisque_score is not None:
        print(f"BRISQUE Score: {brisque_score:.2f}")
    if niqe_score is not None:
        print(f"NIQE Score: {niqe_score:.2f}")
    if ilniqe_score is not None:
        print(f"ILNIQE Score: {ilniqe_score:.2f}")
    if fft_score is not None:
        print(f"FFT Score: {fft_score:.2f}")
    print(f"A imagem foi classificada como: {final_result}")

    # Exibe o tempo de execução
    print(f"Tempo de execução: {execution_time} segundos")

if __name__ == "__main__":
    main()