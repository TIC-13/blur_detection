import tensorflow as tf
import numpy as np 
from PIL import Image
import os

# Pré-processamento
def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert('RGB')
    # Redimensiona a imagem pra a forma esperada pelo modelo
    image = image.resize((input_shape[1], input_shape[2]))
    # Converte a imagem pra um array NumPy de tipo float32
    image = np.array(image, dtype=np.float32)
    # Normaliza os valores da imagem
    image = image / 255.0
    # Adiciona uma dimensão extra pro batch size
    image = np.expand_dims(image, axis=0)
    return image

def apply_model_to_images(image_directory, interpreter):
    # Obtém detalhes das entradas e saídas do modelo
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Obtém a forma esperada da entrada do modelo
    input_shape = input_details[0]['shape']

    results = []

    class_names = ['blurred', 'sharp']

    # Itera sobre todas as imagens no diretório
    for img_file in os.listdir(image_directory):
        image_path = os.path.join(image_directory, img_file)
        
        # Verifica se o arquivo é uma imagem válida
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Arquivo não é uma imagem válida: {image_path}")
            continue
        
        input_data = preprocess_image(image_path, input_shape)
        
        # Define a entrada do modelo
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # Executa a inferência
        interpreter.invoke()
        
        # Obtém a saída do modelo
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Determinando a classe com a maior probabilidade
        predicted_class_idx = np.argmax(output_data)
        predicted_class = class_names[predicted_class_idx]

        probabilities = output_data.tolist()

        result = {
            "img_file": img_file,
            "predicted_class": predicted_class,
            "probabilities": probabilities
        }
        results.append(result)
        
        print(f"Resultados para {img_file}: {result}")

    return results

def main():
    # Caminhos dos arquivos
    model_path = "blur.tflite"
    image_directory = "/images"
    results_path = "modelo_sozinho/results.csv"

    # Carrega o modelo TFLite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Aplica o modelo diretamente às imagens no diretório
    results = apply_model_to_images(image_directory, interpreter)

    # Convertendo os resultados e salvando
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
    print(f"Resultados salvos em {results_path}")

if __name__ == "__main__":
    main()
