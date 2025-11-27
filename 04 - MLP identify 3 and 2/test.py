import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class EditorImagensApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Editor de Dataset 20x20 - Avançado")
        self.root.geometry("1000x750")

        # --- Variáveis de Estado ---
        self.dados_originais = None
        self.indice_atual = 0
        
        # --- Configurações Iniciais ---
        self.var_linhas = tk.IntVar(value=10)        # Linhas a afetar
        self.var_modo = tk.StringVar(value="ruido")  # Modo
        self.var_intensidade = tk.DoubleVar(value=1.0) # Quão forte é o valor do ruído
        self.var_densidade = tk.DoubleVar(value=100.0) # NOVO: Quantos pixeis são afetados (%)

        # --- Layout Principal ---
        
        # 1. Painel de Controlo
        frame_controlo = tk.LabelFrame(root, text="Parâmetros de Alteração", padx=10, pady=10)
        frame_controlo.pack(fill="x", padx=10, pady=5)

        # Linha 1 de controlos
        frame_linha1 = tk.Frame(frame_controlo)
        frame_linha1.pack(fill="x", pady=5)

        # Slider: Linhas
        tk.Label(frame_linha1, text="1. Área (Linhas):").pack(side="left", padx=5)
        self.slider_linhas = tk.Scale(frame_linha1, from_=0, to=20, orient="horizontal", length=150,
                                      variable=self.var_linhas, command=self.atualizar_visualizacao_evento)
        self.slider_linhas.pack(side="left", padx=5)

        # Combo: Modo
        tk.Label(frame_linha1, text="2. Modo:").pack(side="left", padx=5)
        combo_modo = ttk.Combobox(frame_linha1, textvariable=self.var_modo, width=10,
                                  values=["ruido", "preto", "branco"], state="readonly")
        combo_modo.pack(side="left", padx=5)
        combo_modo.bind("<<ComboboxSelected>>", self.atualizar_visualizacao_evento)

        # Linha 2 de controlos (Sliders de Ajuste Fino)
        frame_linha2 = tk.Frame(frame_controlo)
        frame_linha2.pack(fill="x", pady=10)

        # Slider: Densidade (Percentagem de Pixeis afetados)
        tk.Label(frame_linha2, text="3. Densidade (% afetada):").pack(side="left", padx=5)
        self.slider_densidade = tk.Scale(frame_linha2, from_=0, to=100, orient="horizontal", length=200,
                                         variable=self.var_densidade, command=self.atualizar_visualizacao_evento)
        self.slider_densidade.pack(side="left", padx=5)

        # Slider: Intensidade (Força do valor)
        tk.Label(frame_linha2, text="4. Força do Ruído (Valor):").pack(side="left", padx=5)
        self.slider_intensidade = tk.Scale(frame_linha2, from_=0.0, to=1.0, resolution=0.05, orient="horizontal", length=150,
                                           variable=self.var_intensidade, command=self.atualizar_visualizacao_evento)
        self.slider_intensidade.pack(side="left", padx=5)

        # 2. Painel de Arquivo
        frame_arquivo = tk.Frame(root, pady=5, bg="#eeeeee")
        frame_arquivo.pack(fill="x", padx=10)

        btn_carregar = tk.Button(frame_arquivo, text="📂 Carregar CSV Original", command=self.carregar_csv, width=20)
        btn_carregar.pack(side="left", padx=5, pady=5)

        self.lbl_status = tk.Label(frame_arquivo, text="Nenhum arquivo carregado.", bg="#eeeeee")
        self.lbl_status.pack(side="left", padx=10)

        btn_salvar = tk.Button(frame_arquivo, text="💾 Salvar Novo CSV", command=self.salvar_csv, bg="#add8e6", width=20)
        btn_salvar.pack(side="right", padx=5, pady=5)

        # 3. Visualização e Navegação
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8, 4))
        self.fig.patch.set_facecolor('#f0f0f0')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10)

        frame_nav = tk.Frame(root, pady=10)
        frame_nav.pack()
        tk.Button(frame_nav, text="<< Anterior", command=self.imagem_anterior, width=15).pack(side="left", padx=5)
        self.lbl_indice = tk.Label(frame_nav, text="0 / 0", font=("Arial", 12, "bold"))
        self.lbl_indice.pack(side="left", padx=20)
        tk.Button(frame_nav, text="Próxima >>", command=self.imagem_proxima, width=15).pack(side="left", padx=5)

    # --- Lógica Principal ---

    def aplicar_transformacao(self, img_flat):
        # 1. Copiar e converter para 2D
        img_2d = img_flat.reshape(20, 20).copy()
        
        # Ler valores da GUI
        linhas = self.var_linhas.get()
        modo = self.var_modo.get()
        intensidade_valor = self.var_intensidade.get() # Quão forte é o valor (0 a 1)
        densidade_pct = self.var_densidade.get()       # Quantos pixeis mudar (0 a 100)

        if linhas == 0:
            return img_2d

        # 2. Definir a área a ser afetada
        area_alvo = img_2d[0:linhas, :]
        
        # 3. Criar uma MÁSCARA de probabilidade (A Lógica dos 5%)
        # Gera matriz de 0 a 1. Se valor < densidade/100, retorna True (alterar).
        # Ex: se densidade é 5, só os valores < 0.05 ficam True.
        mascara_alteracao = np.random.rand(linhas, 20) < (densidade_pct / 100.0)

        # 4. Aplicar alteração APENAS onde a máscara é True
        if modo == "preto":
            area_alvo[mascara_alteracao] = 0.0
            
        elif modo == "branco":
            area_alvo[mascara_alteracao] = 1.0
            
        elif modo == "ruido":
            # Gera ruído para toda a área, mas aplica só onde a máscara manda
            ruido_gerado = np.random.rand(linhas, 20) * intensidade_valor
            area_alvo[mascara_alteracao] = ruido_gerado[mascara_alteracao]

        return img_2d

    # --- Funções de Suporte (Iguais às anteriores) ---

    def carregar_csv(self):
        caminho = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not caminho: return
        try:
            with open(caminho, 'r') as f:
                reader = csv.reader(f)
                self.dados_originais = []
                for row in reader:
                    if row: self.dados_originais.append(np.array([float(x) for x in row]))
            if not self.dados_originais: return
            self.indice_atual = 0
            self.lbl_status.config(text=f"Arquivo: {caminho.split('/')[-1]}")
            self.atualizar_visualizacao()
        except Exception as e: messagebox.showerror("Erro", str(e))

    def atualizar_visualizacao_evento(self, event=None):
        self.atualizar_visualizacao()

    def atualizar_visualizacao(self):
        if self.dados_originais is None: return
        
        dados_flat = self.dados_originais[self.indice_atual]
        img_modificada = self.aplicar_transformacao(dados_flat)
        img_original = dados_flat.reshape(20, 20)

        self.ax1.clear()
        self.ax1.imshow(img_original, cmap='gray', vmin=0, vmax=1)
        self.ax1.set_title("Original")
        self.ax1.axis('off')

        self.ax2.clear()
        self.ax2.imshow(img_modificada, cmap='gray', vmin=0, vmax=1)
        self.ax2.set_title(f"Resultado (Topo: {self.var_linhas.get()}px | {self.var_densidade.get()}%)")
        self.ax2.axis('off')
        self.canvas.draw()
        self.lbl_indice.config(text=f"{self.indice_atual + 1} / {len(self.dados_originais)}")

    def imagem_proxima(self):
        if self.dados_originais and self.indice_atual < len(self.dados_originais) - 1:
            self.indice_atual += 1
            self.atualizar_visualizacao()

    def imagem_anterior(self):
        if self.dados_originais and self.indice_atual > 0:
            self.indice_atual -= 1
            self.atualizar_visualizacao()

    def salvar_csv(self):
        if self.dados_originais is None: return
        caminho_saida = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not caminho_saida: return
        try:
            with open(caminho_saida, 'w', newline='') as f:
                writer = csv.writer(f)
                for linha_flat in self.dados_originais:
                    writer.writerow(self.aplicar_transformacao(linha_flat).flatten())
            messagebox.showinfo("Sucesso", "Dataset salvo com novas configurações!")
        except Exception as e: messagebox.showerror("Erro", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = EditorImagensApp(root)
    root.mainloop()