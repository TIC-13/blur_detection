# Este código classifica as imagens com base nos thresholds.

import pandas as pd

# Classificação do modelo
def classify_model_probabilities(probabilities):
    return "blurred" if probabilities[0] > probabilities[1] else "sharp"

# Verificando o nível de confiança da classificação
def classify_confidence(probabilities):
    max_confidence = max(probabilities)
    return "Boa confiança" if max_confidence >= 0.90 else "Baixa confiança"

# BRISQUE
def classify_brisque(brisque_score):
    return "low quality" if brisque_score > 40.7 else "good quality"

# NIQE
def classify_niqe(niqe_score):
    return "low quality" if niqe_score > 7.1 else "good quality"

# ILNIQE
def classify_ilniqe(ilniqe_score):
    return "low quality" if ilniqe_score > 91.0 else "good quality"

# Laplacian
def classify_laplacian(laplacian_value):
    return "low quality" if laplacian_value < 29040000 else "good quality"

# Sobel
def classify_sobel(sobel_value):
    return "low quality" if sobel_value < 25.0 else "good quality"

# FFT
def classify_fft(fft_value):
    return "low quality" if fft_value < 134.35 else "good quality"

# Classificações
def overall_classification(model_classification, brisque_class, niqe_class, ilniqe_class, laplacian_class, sobel_class, fft_class):
    score = 0
    score += 1 if model_classification == "sharp" else -1
    score += 1 if brisque_class == "good quality" else -1
    score += 1 if niqe_class == "good quality" else -1
    score += 1 if ilniqe_class == "good quality" else -1
    score += 1 if laplacian_class == "good quality" else -1
    score += 1 if sobel_class == "good quality" else -1
    score += 1 if fft_class == "good quality" else -1
    return "good quality" if score > 0 else "low quality"

# Nosso csv
# Importante: avaliamos de maneira unitária cada csv, então o csv que você vai usar aqui é o que você salvou no final do 2_IQA_thresholds.py
input_csv = "descobrindo_thresholds/IQA_metrics_and_model_results_cropped.csv"
df = pd.read_csv(input_csv)

# Armazena os resultados
classified_results = []

for index, row in df.iterrows():
    img_file = row["img_file"]
    probabilities = eval(row["Model_Probabilities"])
    brisque_score = row["BRISQUE"]
    niqe_score = row["NIQE"]
    ilniqe_score = row["ILNIQE"]
    laplacian_value = row["Laplacian_Value"]
    sobel_value = row["Sobel_Value"]
    fft_value = row["FFT_Value"]

    # Classifica as métricas
    model_classification = classify_model_probabilities(probabilities)
    confidence_classification = classify_confidence(probabilities)
    brisque_classification = classify_brisque(brisque_score)
    niqe_classification = classify_niqe(niqe_score)
    ilniqe_classification = classify_ilniqe(ilniqe_score)
    laplacian_classification = classify_laplacian(laplacian_value)
    sobel_classification = classify_sobel(sobel_value)
    fft_classification = classify_fft(fft_value)

    # Classificação geral
    overall_class = overall_classification(
        model_classification, 
        brisque_classification, 
        niqe_classification, 
        ilniqe_classification, 
        laplacian_classification, 
        sobel_classification, 
        fft_classification
    )

    classified_results.append({
        "img_file": img_file,
        "Model_Probabilities": probabilities,
        "Model_classification": model_classification,
        "Confidence_classification": confidence_classification, 
        "BRISQUE": brisque_score,
        "BRISQUE_classification": brisque_classification,
        "NIQE": niqe_score,
        "NIQE_classification": niqe_classification,
        "ILNIQE": ilniqe_score,
        "ILNIQE_classification": ilniqe_classification,
        "Laplacian": laplacian_value,
        "Laplacian_classification": laplacian_classification,
        "Sobel": sobel_value,
        "Sobel_classification": sobel_classification,
        "FFT": fft_value,
        "FFT_classification": fft_classification,
        "Overall_classification": overall_class
    })

classified_df = pd.DataFrame(classified_results)

output_csv = "descobrindo_thresholds/IQA_classification.csv"
classified_df.to_csv(output_csv, index=False)

print(f"Análises e classificações salvas em {output_csv}")
