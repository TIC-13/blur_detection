# Este código visa aplicar o algoritmo dos 3 juízes em um diretório geral de imagens

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
import argparse
import multiprocessing

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

# Função para calcular a FFT
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

def evaluate_image(image_path, interpreter, ilniqe_model, device):
    prediction, confidence = apply_model_and_check_confidence(image_path, interpreter)
    
    if confidence >= 0.90:
        return prediction, confidence, None, None
    
    ilniqe_score = calculate_ilniqe(image_path, ilniqe_model, device)
    fft_score = calculate_fft(image_path)
    
    ilniqe_quality = classify_ilniqe(ilniqe_score)
    fft_quality = classify_fft(fft_score)
    
    votes = {
        "model": "good quality" if prediction == "sharp" else "low quality",
        "ilniqe": ilniqe_quality,
        "fft": fft_quality
    }
    
    final_classification = "sharp" if list(votes.values()).count("good quality") > 1 else "blurred"
    
    return final_classification, confidence, ilniqe_score, fft_score

def get_arguments():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_directory', type=str, default='../../Data/HAM10000/images')
    parser.add_argument('--output_csv_path', type=str, default='')
    parser.add_argument('--output_txt_path', type=str, default='')
    parser.add_argument('--model_path', type=str, default='../blur.tflite')
    parser.add_argument('--log', action='store_true', default=False, help="if true, save a log file")
    parser.add_argument('--verbose', action='store_true', default=False, help="if true, prints info for each image")
    parser.add_argument('--num_processes', type=int, default=1, help="Number of processes to run in parallel")
    parser.add_argument('--device', type= str, choices= ["cpu", "cuda"], default="cpu")
    
    args = parser.parse_args()
    return args

def get_all_images(image_directory):
    all_images = sorted(os.listdir(image_directory))
    
    # Filter out files that don't look like images if you want, 
    # or just let judge(...) skip them. 
    # For safety, we do minimal filtering here (optional).
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    all_images = [img for img in all_images if img.lower().endswith(valid_extensions)]
    return all_images

def judge(proc_id, interpreter, ilniqe_model, device, args, image_list):
    
    start_time = time.time()
    results = []
    with open(args.output_txt_path, 'w') as txt_file:
        total_images = len(image_list)
        p = 10
        for img_index, image_name in enumerate(image_list):
            pctg = round(100*img_index/total_images)
            if pctg > p:
                print(f"process {proc_id}: {pctg}% done")
                p += 10

            image_path = os.path.join(args.image_directory, image_name)
            
            if not image_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
                continue
            
            if not os.path.exists(image_path):
                if args.verbose: print(f"Imagem não encontrada: {image_path}")
                if args.log: txt_file.write(f"Imagem não encontrada: {image_path}\n")
                continue
            
            final_result, confidence, ilniqe_score, fft_score = evaluate_image(image_path, interpreter, ilniqe_model, device)
            results.append([image_name, final_result, confidence, ilniqe_score, fft_score])
            
            if args.verbose: print(f"Imagem: {image_name}")
            if args.verbose: print(f"Confiança do modelo: {confidence:.2f}")
            if args.log: txt_file.write(f"Imagem: {image_name}\n")
            if args.log: txt_file.write(f"Confiança do modelo: {confidence:.2f}\n")
            if ilniqe_score is not None:
                if args.verbose: print(f"ILNIQE Score: {ilniqe_score:.2f}")
                if args.log: txt_file.write(f"ILNIQE Score: {ilniqe_score:.2f}\n")
            if fft_score is not None:
                if args.verbose: print(f"FFT Score: {fft_score:.2f}")
                if args.log: txt_file.write(f"FFT Score: {fft_score:.2f}\n")
            if args.verbose: print(f"A imagem foi classificada como: {final_result}")
            if args.log: txt_file.write(f"A imagem foi classificada como: {final_result}\n")
            if args.verbose: print("-" * 50)
            if args.log: txt_file.write("-" * 50 + "\n")
        
        execution_time = time.time() - start_time

        if args.verbose: print(f"Tempo de execução total: {execution_time} segundos")
        if args.log: txt_file.write(f"Tempo de execucao total: {execution_time} segundos\n")

    return results

def worker_judge(proc_id, image_list, args, return_dict):
    """
    Each process will re-initialize the TFLite interpreter and ILNIQE model,
    then call judge(...) on its sub-list of images.
    The results get stored in return_dict[proc_id].
    """
    # Re-initialize model and interpreter inside the child process
    device = args.device
    ilniqe_model = iqa.create_metric('ilniqe', device=device)
    
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    
    results = judge(proc_id, interpreter, ilniqe_model, device, args, image_list)
    return_dict[proc_id] = results  # Store results in a shared dict keyed by process ID


def parallel_judge(args):
    """
    This function divides all images into chunks, spawns multiple processes,
    and merges the results from each process.
    """
    all_images = get_all_images(args.image_directory)
    total_images = len(all_images)

    if total_images == 0:
        print("No images found in the directory.")
        return []
    
    num_processes = args.num_processes if args.num_processes > 0 else 1
    
    # Decide how many images per process
    chunk_size = total_images // num_processes
    print("chunk_size:", chunk_size)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    processes = []
    # Launch each process with its sub-list
    for proc_id in range(num_processes):
        start_idx = proc_id * chunk_size
        if proc_id == num_processes-1:
            end_idx = total_images
        else:
            end_idx = min((proc_id + 1) * chunk_size, total_images)
        
        if start_idx >= end_idx:
            break  # No more images
        
        chunk_list = all_images[start_idx:end_idx]
        
        p = multiprocessing.Process(
            target=worker_judge,
            args=(proc_id, chunk_list, args, return_dict)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
     # Combine all partial results
    combined_results = []
    for proc_id in sorted(return_dict.keys()):
        combined_results.extend(return_dict[proc_id])
    
    return combined_results

def main():    
    args= get_arguments()
    
    if not args.output_csv_path:
        args.output_csv_path = os.path.join(os.path.dirname(args.image_directory),"resultados_3_juizes_imagens_anotadas.csv")
    if not args.output_txt_path:
        args.output_txt_path = os.path.join(os.path.dirname(args.image_directory),"resultados_3_juizes_imagens_anotadas.txt")

    # If only 1 process was asked, run single-process approach directly
    # Otherwise, run in parallel and merge results
    if args.num_processes == 1:
        device = args.device
        ilniqe_model = iqa.create_metric('ilniqe', device=device)
        
        interpreter = tf.lite.Interpreter(model_path=args.model_path)
        interpreter.allocate_tensors()
        
        all_images = get_all_images(args.image_directory)
        results = judge(0, interpreter, ilniqe_model, device, args, all_images,)
    else:
        results = parallel_judge(args)
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results, columns=['IMG_FILE', 'FINAL_RESULT', 'CONFIDENCE', 'ILNIQE_SCORE', 'FFT_SCORE'])
        results_df.to_csv(args.output_csv_path, index=False)
        print(f"Results saved to {args.output_csv_path}")
    else:
        print("No results to save. Check your directory.")
    

if __name__ == "__main__":
    main()
