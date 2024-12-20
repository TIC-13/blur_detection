import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, ConfusionMatrixDisplay
import numpy as np

# Nossos resultados
def load_results(csv_path):
    return pd.read_csv(csv_path)

# Métricas de avaliação
def evaluate_results(df, output_path):
    # Labels reais e previstos
    y_true = df['BLURRING'].map({'blurred': 0, 'sharp': 1}).values
    y_pred = df['FINAL_RESULT'].map({'blurred': 0, 'sharp': 1}).values

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Blurred', 'Sharp'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.savefig(f'{output_path}/matriz_confusao_3.png') # renomeie para matriz_confusao_5.png se for o caso
    plt.close()

    # Acurácia
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Acurácia: {accuracy:.2f}')

    # Precisão
    precision = precision_score(y_true, y_pred)
    print(f'Precisão: {precision:.2f}')

    # Recall
    recall = recall_score(y_true, y_pred)
    print(f'Recall: {recall:.2f}')

    # F1-Score
    f1 = f1_score(y_true, y_pred)
    print(f'F1-Score: {f1:.2f}')

    # Quantidade de imagens classificadas como "sharp" e "blurred" de acordo com o resultado final
    sharp_count = np.sum(y_pred == 1)
    blurred_count = np.sum(y_pred == 0)
    print(f'Quantidade de imagens classificadas como "sharp": {sharp_count}')
    print(f'Quantidade de imagens classificadas como "blurred": {blurred_count}')

    # Salvando
    with open(f'{output_path}/resultados_3.txt', 'w') as f: # renomeie para resultados_5.txt se for o caso
        f.write(f'Acurácia: {accuracy:.2f}\n')
        f.write(f'Precisão: {precision:.2f}\n')
        f.write(f'Recall: {recall:.2f}\n')
        f.write(f'F1-Score: {f1:.2f}\n')
        f.write(f'Quantidade de imagens classificadas como "sharp": {sharp_count}\n')
        f.write(f'Quantidade de imagens classificadas como "blurred": {blurred_count}\n')

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.savefig(f'{output_path}/curva_roc_3.png') # renomeie para curva_roc_5.png se for o caso
    plt.close()

# Função principal
def main():
    csv_path = 'imagens_anotadas/resultados_3_juizes/resultados_3_juizes_imagens_anotadas.csv' # Aqui, vc muda pro caminho do seu csv obtido no código anterior
    output_path = 'imagens_anotadas/resultados_3_juizes' # Aqui, vc muda pro caminho onde vc quer salvar as imagens e txt, seja o 3 juízes ou 5 juízes
    df = load_results(csv_path)
    evaluate_results(df, output_path)

if __name__ == "__main__":
    main()
