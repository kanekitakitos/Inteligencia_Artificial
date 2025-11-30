import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Função auxiliar para redimensionar (Zoom) usando apenas Numpy
# Evita a necessidade de instalar bibliotecas extras como scikit-image ou scipy
def redimensionar_imagem_numpy(imagem, escala):
    if escala == 1.0:
        return imagem

    linhas_orig, cols_orig = imagem.shape

    # 1. Calcular novas dimensões
    novas_linhas = int(linhas_orig * escala)
    novas_cols = int(cols_orig * escala)

    # 2. Interpolação "Nearest Neighbor" simples (para manter o estilo pixel art)
    linhas_indices = (np.arange(novas_linhas) * (linhas_orig / novas_linhas)).astype(int)
    cols_indices = (np.arange(novas_cols) * (cols_orig / novas_cols)).astype(int)

    img_redimensionada = imagem[linhas_indices[:, None], cols_indices]

    # 3. Ajustar para voltar a ser 20x20 (Crop ou Padding)
    resultado = np.zeros((20, 20))

    center_r_novo, center_c_novo = novas_linhas // 2, novas_cols // 2
    center_r_orig, center_c_orig = 10, 10  # Centro de 20x20

    # Coordenadas de corte/colagem
    start_r_src = max(0, center_r_novo - 10)
    end_r_src = min(novas_linhas, center_r_novo + 10)
    start_c_src = max(0, center_c_novo - 10)
    end_c_src = min(novas_cols, center_c_novo + 10)

    start_r_dst = max(0, 10 - (center_r_novo - start_r_src))
    end_r_dst = start_r_dst + (end_r_src - start_r_src)
    start_c_dst = max(0, 10 - (center_c_novo - start_c_src))
    end_c_dst = start_c_dst + (end_c_src - start_c_src)

    if (end_r_dst - start_r_dst > 0) and (end_c_dst - start_c_dst > 0):
        resultado[start_r_dst:end_r_dst, start_c_dst:end_c_dst] = \
            img_redimensionada[start_r_src:end_r_src, start_c_src:end_c_src]

    return resultado


class EditorImagensApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Editor Dataset 20x20 - Círculos e Zoom")
        self.root.geometry("1100x850")

        # --- Variáveis de Estado ---
        self.dados_originais = None
        self.indice_atual = 0

        # --- Configurações ---
        self.var_forma = tk.StringVar(value="linear")  # linear ou circular
        self.var_linhas = tk.IntVar(value=10)  # Para modo Linear
        self.var_raio = tk.DoubleVar(value=8.0)  # Para modo Circular

        self.var_zoom = tk.DoubleVar(value=1.0)  # Escala do numero

        self.var_modo = tk.StringVar(value="ruido")  # Tipo de preenchimento
        self.var_intensidade = tk.DoubleVar(value=1.0)
        self.var_densidade = tk.DoubleVar(value=100.0)

        # --- Layout ---

        # 1. Painel de Controlo
        frame_controlo = tk.LabelFrame(root, text="Painel de Transformação", padx=10, pady=10)
        frame_controlo.pack(fill="x", padx=10, pady=5)

        # ---> LINHA 1: Geometria (Forma da área afetada)
        frame_geo = tk.LabelFrame(frame_controlo, text="1. Geometria da Área", padx=5, pady=5)
        frame_geo.pack(fill="x", pady=5)

        # Radio buttons para escolher Linear vs Circular
        tk.Radiobutton(frame_geo, text="Linhas (Topo)", variable=self.var_forma,
                       value="linear", command=self.atualizar_visualizacao_evento).pack(side="left", padx=10)
        tk.Radiobutton(frame_geo, text="Círculo (Foco no Centro)", variable=self.var_forma,
                       value="circular", command=self.atualizar_visualizacao_evento).pack(side="left", padx=10)

        # Separador visual
        ttk.Separator(frame_geo, orient='vertical').pack(side="left", fill='y', padx=10, pady=2)

        # Slider Linhas (só afeta se for Linear)
        tk.Label(frame_geo, text="Qtd Linhas:").pack(side="left")
        self.slider_linhas = tk.Scale(frame_geo, from_=0, to=20, orient="horizontal", length=100,
                                      variable=self.var_linhas, command=self.atualizar_visualizacao_evento)
        self.slider_linhas.pack(side="left", padx=5)

        # Slider Raio (só afeta se for Circular)
        tk.Label(frame_geo, text="Raio do Círculo:").pack(side="left")
        self.slider_raio = tk.Scale(frame_geo, from_=0, to=15, resolution=0.5, orient="horizontal", length=100,
                                    variable=self.var_raio, command=self.atualizar_visualizacao_evento)
        self.slider_raio.pack(side="left", padx=5)

        # ---> LINHA 2: Zoom e Tamanho
        frame_zoom = tk.LabelFrame(frame_controlo, text="2. Tamanho do Conteúdo (Zoom)", padx=5, pady=5)
        frame_zoom.pack(fill="x", pady=5)

        tk.Label(frame_zoom, text="Escala (0.5=Pequeno, 2.0=Grande):").pack(side="left", padx=5)
        self.slider_zoom = tk.Scale(frame_zoom, from_=0.5, to=2.5, resolution=0.1, orient="horizontal", length=300,
                                    variable=self.var_zoom, command=self.atualizar_visualizacao_evento)
        self.slider_zoom.pack(side="left", padx=5)
        tk.Button(frame_zoom, text="Reset Zoom",
                  command=lambda: [self.var_zoom.set(1.0), self.atualizar_visualizacao()],
                  bg="#ddd", height=1).pack(side="left", padx=10)

        # ---> LINHA 3: Efeitos (Ruído/Cores)
        frame_efeito = tk.LabelFrame(frame_controlo, text="3. Tipo de Ruído/Efeito", padx=5, pady=5)
        frame_efeito.pack(fill="x", pady=5)

        tk.Label(frame_efeito, text="Modo:").pack(side="left")
        combo_modo = ttk.Combobox(frame_efeito, textvariable=self.var_modo, width=10,
                                  values=["ruido", "preto", "branco"], state="readonly")
        combo_modo.pack(side="left", padx=5)
        combo_modo.bind("<<ComboboxSelected>>", self.atualizar_visualizacao_evento)

        tk.Label(frame_efeito, text="Densidade (%):").pack(side="left")
        tk.Scale(frame_efeito, from_=0, to=100, orient="horizontal", length=120,
                 variable=self.var_densidade, command=self.atualizar_visualizacao_evento).pack(side="left")

        tk.Label(frame_efeito, text="Intensidade:").pack(side="left")
        tk.Scale(frame_efeito, from_=0.0, to=1.0, resolution=0.05, orient="horizontal", length=120,
                 variable=self.var_intensidade, command=self.atualizar_visualizacao_evento).pack(side="left")

        # 2. Painel de Arquivo
        frame_arquivo = tk.Frame(root, pady=5, bg="#eeeeee")
        frame_arquivo.pack(fill="x", padx=10)

        btn_carregar = tk.Button(frame_arquivo, text="📂 Carregar CSV", command=self.carregar_csv, width=20)
        btn_carregar.pack(side="left", padx=5, pady=5)

        self.lbl_status = tk.Label(frame_arquivo, text="Nenhum arquivo carregado.", bg="#eeeeee")
        self.lbl_status.pack(side="left", padx=10)

        btn_salvar = tk.Button(frame_arquivo, text="💾 Salvar CSV", command=self.salvar_csv, bg="#add8e6", width=20)
        btn_salvar.pack(side="right", padx=5, pady=5)

        # 3. Visualização
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
        self.fig.patch.set_facecolor('#f0f0f0')

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10)

        frame_nav = tk.Frame(root, pady=10)
        frame_nav.pack()
        tk.Button(frame_nav, text="<< Anterior", command=self.imagem_anterior, width=15).pack(side="left", padx=5)
        self.lbl_indice = tk.Label(frame_nav, text="0 / 0", font=("Arial", 12, "bold"))
        self.lbl_indice.pack(side="left", padx=20)
        tk.Button(frame_nav, text="Próxima >>", command=self.imagem_proxima, width=15).pack(side="left", padx=5)

    # --- Lógica de Transformação ---

    def aplicar_transformacao(self, img_flat):
        # 1. Converter para 2D
        img_2d = img_flat.reshape(20, 20).copy()

        # 2. APLICAR ZOOM (Se necessário)
        zoom_fator = self.var_zoom.get()
        if zoom_fator != 1.0:
            img_2d = redimensionar_imagem_numpy(img_2d, zoom_fator)

        # Ler variáveis
        forma = self.var_forma.get()
        modo = self.var_modo.get()
        intensidade = self.var_intensidade.get()
        densidade = self.var_densidade.get()

        # Criar Máscara da Área (True = Onde o ruído será aplicado)
        mask_area = np.zeros((20, 20), dtype=bool)

        if forma == "linear":
            linhas = self.var_linhas.get()
            if linhas > 0:
                mask_area[0:linhas, :] = True

        elif forma == "circular":
            raio = self.var_raio.get()
            # Criar grid de coordenadas
            y, x = np.ogrid[:20, :20]
            centro = 9.5  # Centro do pixel (0-19, o centro é 9.5)
            dist_do_centro = np.sqrt((x - centro) ** 2 + (y - centro) ** 2)

            # Se a distância for MAIOR que o raio, aplicamos o ruído (escondemos o exterior)
            mask_area = dist_do_centro > raio

        # Se não houver área selecionada, retorna imagem (com zoom aplicado)
        if not np.any(mask_area):
            return img_2d

        # 3. Aplicar Densidade (% de pixeis dentro da área afetada)
        # Gera uma máscara aleatória
        mask_densidade = np.random.rand(20, 20) < (densidade / 100.0)

        # Máscara Final = Área Geométrica E Densidade
        mask_final = np.logical_and(mask_area, mask_densidade)

        # 4. Aplicar o efeito na Máscara Final
        if modo == "preto":
            img_2d[mask_final] = 0.0
        elif modo == "branco":
            img_2d[mask_final] = 1.0
        elif modo == "ruido":
            ruido = np.random.rand(20, 20) * intensidade
            img_2d[mask_final] = ruido[mask_final]

        return img_2d

    # --- Funções de Arquivo e Interface (Padrão) ---

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
        except Exception as e:
            messagebox.showerror("Erro", str(e))

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

        txt_info = f"Zoom: {self.var_zoom.get()}x"
        if self.var_forma.get() == "circular":
            txt_info += f" | Círculo R={self.var_raio.get()}"
        else:
            txt_info += f" | Linhas={self.var_linhas.get()}"

        self.ax2.set_title(f"Modificado\n({txt_info})", fontsize=9)
        self.ax2.axis('off')

        # Desenhar circulo vermelho apenas para guia visual se estiver no modo circular
        if self.var_forma.get() == "circular":
            circle = plt.Circle((9.5, 9.5), self.var_raio.get(), color='red', fill=False, linestyle='--', alpha=0.5)
            self.ax2.add_patch(circle)

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
                    # Aplica a transformação atual a todas as imagens ao salvar
                    writer.writerow(self.aplicar_transformacao(linha_flat).flatten())
            messagebox.showinfo("Sucesso", "Dataset salvo com zoom e efeitos!")
        except Exception as e:
            messagebox.showerror("Erro", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = EditorImagensApp(root)
    root.mainloop()