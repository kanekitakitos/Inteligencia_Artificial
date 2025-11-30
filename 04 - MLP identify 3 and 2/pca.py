import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURAÇÃO ---
# Certifique-se de que estes arquivos estão na mesma pasta que o pca.py
arquivo_dados = 'src/data/ruido.csv'   
arquivo_labels = 'src/data/labels.csv' 

def realizar_pca_3d():
    try:
        # 1. Carregar os dados
        df_features = pd.read_csv(arquivo_dados)
        df_labels = pd.read_csv(arquivo_labels)
        print("Arquivos carregados com sucesso.")
        print(f"Dimensões: {df_features.shape}")
        
    except FileNotFoundError:
        print(f"ERRO: Não encontrei os arquivos '{arquivo_dados}' ou '{arquivo_labels}'.")
        print("Verifique se o nome está correto e se estão na pasta:", 
              r"C:\Users\brand\OneDrive\Documentos\GitHub\Inteligencia_Artificial\04 - MLP identify 3 and 2")
        return

    # 2. Padronização
    features = df_features.values
    x = StandardScaler().fit_transform(features)

    # 3. Aplicar PCA (3 componentes)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)

    # DataFrame com 3 colunas
    principalDf = pd.DataFrame(data=principalComponents, 
                               columns=['PC1', 'PC2', 'PC3'])

    # 4. Juntar com Labels
    # Pega a primeira coluna do arquivo de labels como alvo
    principalDf['Target'] = df_labels.iloc[:, 0].values

    # 5. Visualização 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title('PCA em 3 Dimensões', fontsize=15)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')

    targets = principalDf['Target'].unique()
    
    for target in targets:
        indicesToKeep = principalDf['Target'] == target
        ax.scatter(principalDf.loc[indicesToKeep, 'PC1'], 
                   principalDf.loc[indicesToKeep, 'PC2'], 
                   principalDf.loc[indicesToKeep, 'PC3'], 
                   s=50, 
                   label=target)

    ax.legend()
    ax.grid(True)
    
    print("Gráfico gerado. A janela deve abrir em instantes.")
    plt.show()

if __name__ == "__main__":
    realizar_pca_3d()