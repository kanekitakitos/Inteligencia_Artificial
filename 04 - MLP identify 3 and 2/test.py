import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Função Auxiliar de Zoom ---
def redimensionar_imagem_numpy(imagem, escala):
    if escala == 1.0: return imagem
    linhas_orig, cols_orig = imagem.shape
    novas_linhas = int(linhas_orig * escala)
    novas_cols = int(cols_orig * escala)

    linhas_indices = (np.arange(novas_linhas) * (linhas_orig / novas_linhas)).astype(int)
    cols_indices = (np.arange(novas_cols) * (cols_orig / novas_cols)).astype(int)
    img_redimensionada = imagem[linhas_indices[:, None], cols_indices]

    resultado = np.zeros((20, 20))
    center_r_novo, center_c_novo = novas_linhas // 2, novas_cols // 2

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
        self.root.title("Editor Dataset 20x20 - Navegação Rápida")
        self.root.geometry("1200x950") # Aumentei um pouco a altura

        # --- Estado ---
        self.dados_originais = None
        self.indice_atual = 0

        # --- Variáveis ---
        self.var_forma = tk.StringVar(value="linear")
        self.var_linhas = tk.IntVar(value=10)
        self.var_raio = tk.DoubleVar(value=8.0)
        self.var_zoom = tk.DoubleVar(value=1.0)
        self.var_modo = tk.StringVar(value="ruido")
        self.var_intensidade = tk.DoubleVar(value=1.0)
        self.var_densidade = tk.DoubleVar(value=100.0)

        # Grid Vars
        self.var_grid_size = tk.IntVar(value=2)
        self.grid_vars = []
        self.frame_checkboxes = None

        # ================= LAYOUT =================

        # 1. Painel de Controlo
        frame_controlo = tk.LabelFrame(root, text="Painel de Transformação", padx=10, pady=10)
        frame_controlo.pack(fill="x", padx=10, pady=5)

        # ---> LINHA 1: Geometria
        frame_geo = tk.LabelFrame(frame_controlo, text="1. Geometria e Seleção", padx=5, pady=5)
        frame_geo.pack(fill="x", pady=5)

        frame_radios = tk.Frame(frame_geo)
        frame_radios.pack(side="left", padx=5, anchor="n")
        tk.Radiobutton(frame_radios, text="Linhas (Topo)", variable=self.var_forma, value="linear", command=self.toggle_interface).pack(anchor="w")
        tk.Radiobutton(frame_radios, text="Círculo (Centro)", variable=self.var_forma, value="circular", command=self.toggle_interface).pack(anchor="w")
        tk.Radiobutton(frame_radios, text="Grid (Matriz)", variable=self.var_forma, value="grid", command=self.toggle_interface).pack(anchor="w")

        ttk.Separator(frame_geo, orient='vertical').pack(side="left", fill='y', padx=15, pady=2)
        self.frame_params = tk.Frame(frame_geo)
        self.frame_params.pack(side="left", fill="both", expand=True)

        self.construir_interface_grid()
        self.toggle_interface()

        # ---> LINHA 2: Zoom
        frame_zoom = tk.LabelFrame(frame_controlo, text="2. Zoom", padx=5, pady=5)
        frame_zoom.pack(fill="x", pady=5)
        tk.Label(frame_zoom, text="Escala:").pack(side="left")
        tk.Scale(frame_zoom, from_=0.5, to=2.5, resolution=0.1, orient="horizontal", length=250, variable=self.var_zoom, command=self.atualizar_visualizacao_evento).pack(side="left", padx=5)

        # ---> LINHA 3: Efeitos
        frame_efeito = tk.LabelFrame(frame_controlo, text="3. Efeitos", padx=5, pady=5)
        frame_efeito.pack(fill="x", pady=5)
        ttk.Combobox(frame_efeito, textvariable=self.var_modo, values=["ruido", "preto", "branco"], width=10, state="readonly").pack(side="left", padx=5)
        tk.Label(frame_efeito, text="Densidade (%):").pack(side="left", padx=(10,0))
        tk.Scale(frame_efeito, from_=0, to=100, variable=self.var_densidade, orient="horizontal", command=self.atualizar_visualizacao_evento).pack(side="left")
        tk.Label(frame_efeito, text="Força:").pack(side="left", padx=(10,0))
        tk.Scale(frame_efeito, from_=0.0, to=1.0, resolution=0.05, variable=self.var_intensidade, orient="horizontal", command=self.atualizar_visualizacao_evento).pack(side="left")

        # 2. Arquivo
        frame_arquivo = tk.Frame(root, bg="#eee", pady=5)
        frame_arquivo.pack(fill="x", padx=10)
        tk.Button(frame_arquivo, text="📂 Carregar CSV", command=self.carregar_csv).pack(side="left", padx=5)
        self.lbl_status = tk.Label(frame_arquivo, text="Sem arquivo", bg="#eee")
        self.lbl_status.pack(side="left", padx=10)
        tk.Button(frame_arquivo, text="💾 Salvar CSV", command=self.salvar_csv, bg="#b3e5fc").pack(side="right", padx=5)

        # 3. Plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8, 4))
        self.fig.patch.set_facecolor('#f0f0f0')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10)

        # 4. NAVEGAÇÃO (ATUALIZADA)

        # Frame de Saltos Rápidos
        frame_saltos = tk.Frame(root, pady=5)
        frame_saltos.pack()

        tk.Button(frame_saltos, text="⏮ Início (0)", command=self.ir_para_inicio, width=12, bg="#e1f5fe").pack(side="left", padx=5)
        tk.Button(frame_saltos, text="🎯 Meio (Os 3s)", command=self.ir_para_meio, width=15, bg="#fff9c4").pack(side="left", padx=5)
        tk.Button(frame_saltos, text="⏭ Fim", command=self.ir_para_fim, width=12, bg="#e1f5fe").pack(side="left", padx=5)

        tk.Label(frame_saltos, text="| Ir para:").pack(side="left", padx=5)
        self.entry_ir = tk.Entry(frame_saltos, width=6)
        self.entry_ir.pack(side="left")
        self.entry_ir.bind('<Return>', self.ir_para_indice_manual) # Permite dar Enter

        # Frame de Setas (Passo a Passo)
        frame_nav = tk.Frame(root, pady=5)
        frame_nav.pack()
        tk.Button(frame_nav, text="<< Anterior", command=self.imagem_anterior, width=15).pack(side="left", padx=5)
        self.lbl_indice = tk.Label(frame_nav, text="0 / 0", font=("Arial", 12, "bold"))
        self.lbl_indice.pack(side="left", padx=20)
        tk.Button(frame_nav, text="Próxima >>", command=self.imagem_proxima, width=15).pack(side="left", padx=5)

    # ================= NOVAS FUNÇÕES DE NAVEGAÇÃO =================

    def ir_para_inicio(self):
        if self.dados_originais:
            self.indice_atual = 0
            self.atualizar_visualizacao()

    def ir_para_meio(self):
        if self.dados_originais:
            # Pega o tamanho total e divide por 2 (divisão inteira)
            meio = len(self.dados_originais) // 2
            self.indice_atual = meio
            self.atualizar_visualizacao()

    def ir_para_fim(self):
        if self.dados_originais:
            self.indice_atual = len(self.dados_originais) - 1
            self.atualizar_visualizacao()

    def ir_para_indice_manual(self, event=None):
        if not self.dados_originais: return
        try:
            val = int(self.entry_ir.get())
            # Ajustar para base 0 (humano escreve 1, computador lê 0)
            idx = val - 1
            if 0 <= idx < len(self.dados_originais):
                self.indice_atual = idx
                self.atualizar_visualizacao()
            else:
                messagebox.showwarning("Aviso", "Número fora do limite!")
        except ValueError:
            pass # Ignora se não for número

    # ================= LOGICA GRID/INTERFACE (Mantida) =================

    def construir_interface_grid(self):
        for widget in self.frame_params.winfo_children(): widget.destroy()
        forma = self.var_forma.get()
        if forma == "linear":
            tk.Label(self.frame_params, text="Qtd Linhas:").pack(side="left")
            tk.Scale(self.frame_params, from_=0, to=20, orient="horizontal", variable=self.var_linhas, command=self.atualizar_visualizacao_evento).pack(side="left")
        elif forma == "circular":
            tk.Label(self.frame_params, text="Raio:").pack(side="left")
            tk.Scale(self.frame_params, from_=0, to=15, resolution=0.5, orient="horizontal", variable=self.var_raio, command=self.atualizar_visualizacao_evento).pack(side="left")
        elif forma == "grid":
            f_config = tk.Frame(self.frame_params)
            f_config.pack(side="top", fill="x", pady=(0, 5))
            tk.Label(f_config, text="Grid Size:").pack(side="left")
            self.scale_grid = tk.Scale(f_config, from_=2, to=5, orient="horizontal", length=80, variable=self.var_grid_size, command=self.gerar_grid_checkboxes)
            self.scale_grid.pack(side="left", padx=5)
            tk.Button(f_config, text="Todos", command=self.selecionar_tudo_grid, font=("Arial", 8)).pack(side="left", padx=5)
            tk.Button(f_config, text="Limpar", command=self.limpar_tudo_grid, font=("Arial", 8)).pack(side="left")
            self.frame_checkboxes = tk.Frame(self.frame_params, relief="sunken", borderwidth=1)
            self.frame_checkboxes.pack(side="top")
            self.gerar_grid_checkboxes()

    def gerar_grid_checkboxes(self, event=None):
        val = self.var_grid_size.get()
        if val == 3: val = 4
        self.var_grid_size.set(val)
        for widget in self.frame_checkboxes.winfo_children(): widget.destroy()
        self.grid_vars = []
        tamanho = self.var_grid_size.get()
        for r in range(tamanho):
            row_vars = []
            for c in range(tamanho):
                var = tk.BooleanVar(value=False)
                tk.Checkbutton(self.frame_checkboxes, variable=var, command=self.atualizar_visualizacao_evento).grid(row=r, column=c)
                row_vars.append(var)
            self.grid_vars.append(row_vars)
        self.atualizar_visualizacao()

    def selecionar_tudo_grid(self):
        for row in self.grid_vars:
            for var in row: var.set(True)
        self.atualizar_visualizacao()

    def limpar_tudo_grid(self):
        for row in self.grid_vars:
            for var in row: var.set(False)
        self.atualizar_visualizacao()

    def toggle_interface(self):
        self.construir_interface_grid()
        self.atualizar_visualizacao()

    # ================= LOGICA TRANSFORMAÇÃO (Mantida) =================

    def aplicar_transformacao(self, img_flat):
        img_2d = img_flat.reshape(20, 20).copy()
        if self.var_zoom.get() != 1.0: img_2d = redimensionar_imagem_numpy(img_2d, self.var_zoom.get())
        forma = self.var_forma.get()
        mask_area = np.zeros((20, 20), dtype=bool)

        if forma == "linear": mask_area[0:self.var_linhas.get(), :] = True
        elif forma == "circular":
            y, x = np.ogrid[:20, :20]
            mask_area = np.sqrt((x-9.5)**2 + (y-9.5)**2) > self.var_raio.get()
        elif forma == "grid":
            n = self.var_grid_size.get(); step = 20 // n
            for r in range(n):
                for c in range(n):
                    if self.grid_vars[r][c].get(): mask_area[r*step:(r+1)*step, c*step:(c+1)*step] = True

        if not np.any(mask_area): return img_2d

        dens = self.var_densidade.get() / 100.0
        mask_final = np.logical_and(mask_area, np.random.rand(20,20) < dens)
        modo = self.var_modo.get()
        if modo == "preto": img_2d[mask_final] = 0.0
        elif modo == "branco": img_2d[mask_final] = 1.0
        else: img_2d[mask_final] = (np.random.rand(20,20) * self.var_intensidade.get())[mask_final]
        return img_2d

    # ================= VISUALIZAÇÃO E ARQUIVO =================

    def atualizar_visualizacao_evento(self, e=None): self.atualizar_visualizacao()

    def atualizar_visualizacao(self):
        if self.dados_originais is None: return
        dados = self.dados_originais[self.indice_atual]
        img_mod = self.aplicar_transformacao(dados)

        self.ax1.clear(); self.ax1.axis('off'); self.ax1.set_title(f"Original (Idx: {self.indice_atual+1})")
        self.ax1.imshow(dados.reshape(20,20), cmap='gray', vmin=0, vmax=1)

        self.ax2.clear(); self.ax2.axis('off'); self.ax2.set_title("Modificado")
        self.ax2.imshow(img_mod, cmap='gray', vmin=0, vmax=1)

        if self.var_forma.get() == "grid":
            n = self.var_grid_size.get(); step = 20 / n
            for i in range(1, n):
                self.ax2.axhline(y=i*step - 0.5, color='red', alpha=0.3, linestyle='--')
                self.ax2.axvline(x=i*step - 0.5, color='red', alpha=0.3, linestyle='--')
        self.canvas.draw()
        self.lbl_indice.config(text=f"{self.indice_atual + 1} / {len(self.dados_originais)}")

    def carregar_csv(self):
        fn = filedialog.askopenfilename()
        if fn:
            with open(fn, 'r') as f: self.dados_originais = [np.array([float(x) for x in r]) for r in csv.reader(f) if r]
            self.lbl_status.config(text=fn.split('/')[-1])
            self.indice_atual = 0; self.atualizar_visualizacao()

    def salvar_csv(self):
        fn = filedialog.asksaveasfilename(defaultextension=".csv")
        if fn and self.dados_originais:
            with open(fn, 'w', newline='') as f:
                w = csv.writer(f)
                for d in self.dados_originais: w.writerow(self.aplicar_transformacao(d).flatten())
            messagebox.showinfo("OK", "Salvo!")

    def imagem_proxima(self):
        if self.dados_originais and self.indice_atual < len(self.dados_originais)-1:
            self.indice_atual += 1; self.atualizar_visualizacao()
    def imagem_anterior(self):
        if self.dados_originais and self.indice_atual > 0:
            self.indice_atual -= 1; self.atualizar_visualizacao()

if __name__ == "__main__":
    root = tk.Tk()
    app = EditorImagensApp(root)
    root.mainloop()