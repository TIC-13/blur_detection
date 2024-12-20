# Este código faz uma análise da classificação das imagens

import pandas as pd

# Nosso csv
input_csv = "descobrindo_thresholds/IQA_classification.csv"
df = pd.read_csv(input_csv)

# Dicionário pra receber as métricas sem classificar por tipo
analysis_results = {
    "Boa_confiança_e_sharp": 0,
    "Boa_confiança_e_blurred": 0,
    "Baixa_confiança_e_sharp": 0,
    "Baixa_confiança_e_blurred": 0,
    "good_quality_brisque": 0,
    "low_quality_brisque": 0,
    "good_quality_niqe": 0,
    "low_quality_niqe": 0,
    "good_quality_ilniqe": 0,
    "low_quality_ilniqe": 0,
    "good_quality_laplacian": 0,
    "low_quality_laplacian": 0,
    "good_quality_sobel": 0,
    "low_quality_sobel": 0,
    "good_quality_fft": 0,
    "low_quality_fft": 0,
    "good_quality_overall": 0,
    "low_quality_overall": 0
}

# Analisando os dados
for index, row in df.iterrows():
    # Verifica a confiança e classificação do modelo
    confidence_class = row["Confidence_classification"]
    model_class = row["Model_classification"]
    
    if confidence_class == "Boa confiança":
        if model_class == "sharp":
            analysis_results["Boa_confiança_e_sharp"] += 1
        else:
            analysis_results["Boa_confiança_e_blurred"] += 1
    else:
        if model_class == "sharp":
            analysis_results["Baixa_confiança_e_sharp"] += 1
        else:
            analysis_results["Baixa_confiança_e_blurred"] += 1
    
    # Verifica as classificações de qualidade em cada métrica
    brisque_class = row["BRISQUE_classification"]
    niqe_class = row["NIQE_classification"]
    ilniqe_class = row["ILNIQE_classification"]
    laplacian_class = row["Laplacian_classification"]
    sobel_class = row["Sobel_classification"]
    fft_class = row["FFT_classification"]
    overall_class = row["Overall_classification"]
    
    # Atualizando as contagens
    if brisque_class == "good quality":
        analysis_results["good_quality_brisque"] += 1
    else:
        analysis_results["low_quality_brisque"] += 1

    if niqe_class == "good quality":
        analysis_results["good_quality_niqe"] += 1
    else:
        analysis_results["low_quality_niqe"] += 1

    if ilniqe_class == "good quality":
        analysis_results["good_quality_ilniqe"] += 1
    else:
        analysis_results["low_quality_ilniqe"] += 1

    if laplacian_class == "good quality":
        analysis_results["good_quality_laplacian"] += 1
    else:
        analysis_results["low_quality_laplacian"] += 1

    if sobel_class == "good quality":
        analysis_results["good_quality_sobel"] += 1
    else:
        analysis_results["low_quality_sobel"] += 1

    if fft_class == "good quality":
        analysis_results["good_quality_fft"] += 1
    else:
        analysis_results["low_quality_fft"] += 1
    
    # Atualiza a contagem para a classificação geral
    if overall_class == "good quality":
        analysis_results["good_quality_overall"] += 1
    else:
        analysis_results["low_quality_overall"] += 1

# Printando os resultados
print("\nResultados:")
print(f"Boa confiança e sharp: {analysis_results['Boa_confiança_e_sharp']}")
print(f"Boa confiança e blurred: {analysis_results['Boa_confiança_e_blurred']}")
print(f"Baixa confiança e sharp: {analysis_results['Baixa_confiança_e_sharp']}")
print(f"Baixa confiança e blurred: {analysis_results['Baixa_confiança_e_blurred']}")
print(f"BRISQUE - Good quality: {analysis_results['good_quality_brisque']}, Low quality: {analysis_results['low_quality_brisque']}")
print(f"NIQE - Good quality: {analysis_results['good_quality_niqe']}, Low quality: {analysis_results['low_quality_niqe']}")
print(f"ILNIQE - Good quality: {analysis_results['good_quality_ilniqe']}, Low quality: {analysis_results['low_quality_ilniqe']}")
print(f"Laplacian - Good quality: {analysis_results['good_quality_laplacian']}, Low quality: {analysis_results['low_quality_laplacian']}")
print(f"Sobel - Good quality: {analysis_results['good_quality_sobel']}, Low quality: {analysis_results['low_quality_sobel']}")
print(f"FFT - Good quality: {analysis_results['good_quality_fft']}, Low quality: {analysis_results['low_quality_fft']}")
print(f"Overall classification - Good quality: {analysis_results['good_quality_overall']}, Low quality: {analysis_results['low_quality_overall']}")
