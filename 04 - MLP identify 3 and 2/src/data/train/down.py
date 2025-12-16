import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def processar_dataset_normalizado():
    print("--- A iniciar o processo (com normalização) ---")

    # 1. Carregar o dataset MNIST completo
    print("1. A descarregar o dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Juntar treino e teste
    X_todos = np.concatenate((x_train, x_test))
    y_todos = np.concatenate((y_train, y_test))

    # 2. Filtrar apenas os números 2 e 3
    print("2. A filtrar apenas os números 2 e 3...")
    mask = (y_todos == 2) | (y_todos == 3)
    X_filtrado = X_todos[mask]
    y_filtrado = y_todos[mask]

    print(f"   Imagens encontradas: {X_filtrado.shape[0]}")

    # 3. Redimensionar e NORMALIZAR
    print("3. A redimensionar (20x20) e normalizar (0.0 a 1.0)...")
    
    imagens_processadas = []
    
    for img in X_filtrado:
        # Redimensionar usando INTER_AREA (melhor para reduzir tamanho)
        img_resized = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
        
        # --- AQUI ESTÁ A MUDANÇA ---
        # Converter para float e dividir por 255 para ficar entre 0 e 1
        img_norm = img_resized.astype('float32') / 255.0
        
        # Aplainar (flatten) para vetor de 400 posições
        imagens_processadas.append(img_norm.flatten())

    # Converter para array numpy
    dados_finais = np.array(imagens_processadas)

    # 4. Guardar em CSV
    print("4. A guardar os ficheiros CSV...")
    
    df_dados = pd.DataFrame(dados_finais)
    df_labels = pd.DataFrame(y_filtrado)

    # Guardar dataset.csv (valores float)
    # float_format='%.6f' garante que guardamos com precisão de 6 casas decimais
    dataset_filename = 'dataset_2_3_normalized.csv'
    df_dados.to_csv(dataset_filename, index=False, header=False, float_format='%.6f')
    
    # Guardar labels.csv
    labels_filename = 'labels_2_3.csv'
    df_labels.to_csv(labels_filename, index=False, header=False)

    print("--- Sucesso! ---")
    print(f"Dataset normalizado salvo em: {dataset_filename}")
    print("Os valores agora estão entre 0.000000 e 1.000000")

    # 5. Verificação rápida
    print(f"Exemplo do primeiro pixel da primeira imagem: {dados_finais[0][0]}")
    print(f"Valor máximo encontrado no dataset: {np.max(dados_finais)}")

if __name__ == "__main__":
    processar_dataset_normalizado()