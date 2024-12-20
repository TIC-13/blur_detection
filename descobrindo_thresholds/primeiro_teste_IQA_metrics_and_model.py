import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import torch
from torchvision import transforms
import pyiqa as iqa

""" 
Este código contém a aplicação do modelo de classificação de Blur desenvolvido 
pelo time da Lux.AI, da Universidade Federal de Pernambuco em parceria com a Softex, aplicado no HC-UFPE.
Além disso, ele também conta com outras métricas de IQA, como o BRISQUE, NIQE, ILNIQE e a variação do laplaciano.
"""

# Pré-processa a imagem para a aplicação do modelo
def preprocess_image_for_model(image_path, input_shape):
    image = Image.open(image_path).convert('RGB') 
    image = image.resize((input_shape[1], input_shape[2])) # Redimensiona para as dimensões de altura e largura input_shape[1] e input_shape[2]
    image = np.array(image, dtype=np.float32) / 255.0 # Converte a imagem em um array NumPy com tipo de dado float32 e normaliza os valores dos pixels
    return np.expand_dims(image, axis=0) # Adiciona uma dimensão extra ao array para representar o batch size, que é exigido pra entrada do modelo

# Função para aplicação do modelo
def apply_model_to_images(image_list, image_directory, interpreter):
    # Recebe os detalhes dos tensores de entrada e saída e o shape
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    # Armazena os resultados das imagens
    results = []

    for img_file in image_list:
        image_path = os.path.join(image_directory, img_file)
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
    
        input_data = preprocess_image_for_model(image_path, input_shape)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        probabilities = output_data.tolist()

        results.append({"img_file": img_file, "Model_Probabilities": probabilities})
        print(f"Imagem processada: {img_file}")
    return results

"""
O Laplaciano é um operador diferencial que calcula a segunda derivada de uma imagem. 
A variação do Laplaciano mede a quantidade de variação nas intensidades dos pixels da imagem. 
Em imagens nítidas, o Laplaciano tende a ter uma alta variação devido aos detalhes e bordas distintas. 
Em imagens desfocadas, a variação do Laplaciano é menor.

Esta função é baseada no livro de Sandipan Dey.
"""

# Função ajustada para calcular o Laplaciano
def calculate_var_laplacian(image_path):
    image = Image.open(image_path).convert('L')  # Converte a imagem para escala de cinza
    gray_image = np.array(image)  # Converte a imagem para um array NumPy
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()  # Calcula o Laplaciano. cv2.CV_64F especifica a profundidade do tipo de dado como float64
    return laplacian_var

# BRISQUE, NIQE e ILNIQE são métricas de qualidade de imagem sem referência. Aplicação com base na biblioteca PYIQA

"""
Natural Image Quality Evaluator

O NIQE mede a "naturalidade" da imagem, ou seja, quão semelhante a uma imagem natural e limpa ela é.
Ele utiliza um modelo estatístico treinado em um conjunto de imagens naturais e 
calcula a distância entre a distribuição de características da imagem testada e das imagens naturais.
Valores mais baixos indicam maior qualidade.

Improved Local Natural Image Quality Evaluator

O ILNIQE aprimora o NIQE considerando informações locais da imagem em vez de apenas características globais.
Ele faz isso aplicando um modelo de qualidade de imagem a regiões locais da imagem e depois combina 
essas avaliações locais para obter uma avaliação global.
"""

# Funções para calcular NIQE, ILNIQE
def calculate_niqe_ilniqe(image_path, niqe_model, ilniqe_model, device):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))]) # Converte para um tensor PyTorch e redimensiona
    img_tensor = transform(image).unsqueeze(0).to(device) # Aplica transformações, adiciona a dimensão extra do batch size e move o tensor para GPU ou CPU

    niqe_score = niqe_model(img_tensor).item() # .item() serve para converter o tensor de um único valor para um número Python
    ilniqe_score = ilniqe_model(img_tensor).item()
    return niqe_score, ilniqe_score

"""
Blind/Referenceless Image Spatial Quality Evaluator

O BRISQUE usa uma abordagem de modelagem estatística baseada em características de textura.
As características são extraídas e comparadas com um modelo de qualidade treinado. 
O valor resultante representa a degradação percebida da imagem, onde valores mais altos indicam pior qualidade.
"""

# Função para calcular o BRISQUE
def calculate_brisque(image_path, brisque_model, device):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    score = brisque_model(img_tensor).item()
    return score

### Início do processamento ###

def process_images(image_list, image_directory, interpreter, brisque_model, niqe_model, ilniqe_model, device):
    results = []

    # Obtém probabilidades do modelo
    model_results = apply_model_to_images(image_list, image_directory, interpreter)

    for result in model_results:
        img_file = result["img_file"]
        image_path = os.path.join(image_directory, img_file)

        print(f"Processando imagem: {img_file}")

        # Calcula métricas
        brisque_score = calculate_brisque(image_path, brisque_model, device)
        niqe_score, ilniqe_score = calculate_niqe_ilniqe(image_path, niqe_model, ilniqe_model, device)
        laplacian_value = calculate_var_laplacian(image_path)

        # Adiciona os resultados ao resultado final
        results.append({
            "img_file": img_file,
            "Model_Probabilities": result["Model_Probabilities"],
            "BRISQUE": brisque_score,
            "NIQE": niqe_score,
            "ILNIQE": ilniqe_score,
            "Laplacian": laplacian_value
        })

        print(f"Imagem: {img_file} processada!")
    return results

def main():
    # Caminho do arquivo CSV e diretório de imagens
    csv_path = "lesions.csv"
    image_directory = "images"
    model_path = "blur.tflite"
    output_csv = "descobrindo_thresholds/IQA_metrics_and_model_results.csv"

    # Carrega o arquivo CSV e extrai a 10ª coluna (índice 9) com os nomes das imagens
    df = pd.read_csv(csv_path)
    image_list = df.iloc[:, 9].dropna().tolist()

    image_list = [str(img_file) for img_file in image_list]

    # Inicializa o modelo mobilenet
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Inicializa modelos de IQA (BRISQUE, NIQE, ILNIQE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    niqe_model = iqa.create_metric('niqe', device=device)
    ilniqe_model = iqa.create_metric('ilniqe', device=device)
    brisque_model = iqa.create_metric('brisque', device=device)

    # Processa imagens
    results = process_images(image_list, image_directory, interpreter, brisque_model, niqe_model, ilniqe_model, device)

    # Converte para um DataFrame e salva no CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Resultados salvos em {output_csv}")

if __name__ == "__main__":
    main()
