#!/usr/bin/env python
# coding: utf-8

import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image
import os
import threading
# Importa le funzioni di analisi dai moduli specifici
from cell_preprocessing import preprocess_image
from cell_detector import CellDetector


class App(tk.Tk):
    """
    Classe principale dell'applicazione GUI con Tkinter.
    """
    def __init__(self):
        super().__init__()
        self.title("Avian - Analisi Immagini Vetrini")
        self.geometry("800x600")

        # Carica il modello di IA all'avvio dell'app
        try:
            # Rileva il provider di esecuzione migliore per ONNX Runtime
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            provider = 'CPUExecutionProvider' # Default
            if 'CUDAExecutionProvider' in available_providers:
                provider = 'CUDAExecutionProvider' # Usa NVIDIA GPU
            elif 'DmlExecutionProvider' in available_providers:
                provider = 'DmlExecutionProvider' # Usa DirectML su Windows (AMD/Intel GPU)

            self.detector = CellDetector(provider=provider)
        except Exception as e:
            messagebox.showerror("Errore Modello", f"Impossibile caricare il modello ONNX. Assicurati che 'efficientNet_B0.onnx' sia nella cartella.\n\nDettagli: {e}")
            self.destroy()
            return

        # Variabili di stato
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.do_preprocessing = tk.BooleanVar(value=True)
        self.do_counting = tk.BooleanVar(value=True)

        # --- Layout ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Sezione Input
        input_frame = ttk.LabelFrame(main_frame, text="1. Seleziona Immagine Input", padding="10")
        input_frame.pack(fill="x", pady=5)
        ttk.Entry(input_frame, textvariable=self.input_path).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(input_frame, text="Sfoglia...", command=self.browse_input).pack(side="left")

        # Sezione Fasi di Analisi
        steps_frame = ttk.LabelFrame(main_frame, text="2. Seleziona Fasi di Analisi", padding="10")
        steps_frame.pack(fill="x", pady=5)
        ttk.Checkbutton(steps_frame, text="Preprocessing (Isolamento campione)", variable=self.do_preprocessing).pack(anchor="w")
        ttk.Checkbutton(steps_frame, text="Conteggio Cellule (Segmentazione e Bounding Box)", variable=self.do_counting).pack(anchor="w")

        # Sezione Output
        output_frame = ttk.LabelFrame(main_frame, text="3. Salva Immagine Risultato", padding="10")
        output_frame.pack(fill="x", pady=5)
        ttk.Entry(output_frame, textvariable=self.output_path).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(output_frame, text="Salva come...", command=self.browse_output).pack(side="left")

        # Pulsante Avvio
        self.run_button = ttk.Button(main_frame, text="Avvia Analisi", command=self.start_analysis_thread)
        self.run_button.pack(fill="x", pady=10)

        # Log Area
        log_frame = ttk.LabelFrame(main_frame, text="Log e Risultati", padding="10")
        log_frame.pack(fill="both", expand=True, pady=5)
        self.log_text = tk.Text(log_frame, height=10, state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        self.update_idletasks()

    def browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.tif *.heic")])
        if path:
            self.input_path.set(path)
            # Propone un nome di output di default
            base, ext = os.path.splitext(path)
            self.output_path.set(f"{base}_analizzato.png")

    def browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if path:
            self.output_path.set(path)

    def start_analysis_thread(self):
        # Esegue l'analisi in un thread separato per non bloccare la GUI
        self.run_button.config(state="disabled")
        self.log("--- Avvio analisi ---")
        thread = threading.Thread(target=self.run_analysis)
        thread.start()

    def run_analysis(self):
        input_p = self.input_path.get()
        output_p = self.output_path.get()

        if not input_p or not output_p:
            messagebox.showerror("Errore", "Seleziona un file di input e un percorso di output.")
            self.run_button.config(state="normal")
            return

        try:
            self.log(f"1. Caricamento immagine da: {os.path.basename(input_p)}")
            current_image = np.array(Image.open(input_p))

            # Fase di Preprocessing
            if self.do_preprocessing.get():
                self.log("2. Esecuzione del preprocessing...")
                current_image = preprocess_image(current_image)
                self.log("   -> Preprocessing completato.")
            else:
                self.log("2. Preprocessing saltato.")

            # Fase di Conteggio
            if self.do_counting.get():
                self.log(f"3. Esecuzione conteggio (provider: {self.detector.session.get_providers()[0]})...")
                current_image, coco_data = self.detector.detect(current_image)
                counts = coco_data.pop('summary_counts', {})  # Rimuove il conteggio custom e lo salva
                self.log(f"   -> Conteggio completato. Risultati: {counts}")

                # Salva il file JSON
                json_path = os.path.splitext(output_p)[0] + ".json"

                # Aggiorna il nome del file nell'output COCO
                coco_data['images'][0]['file_name'] = os.path.basename(output_p)

                with open(json_path, 'w') as f:
                    json.dump(coco_data, f, indent=4)
                self.log(f"   -> Annotazioni salvate in: {os.path.basename(json_path)}")
            else:
                self.log("3. Conteggio cellule saltato.")

            # Salvataggio
            self.log(f"4. Salvataggio immagine risultato in: {os.path.basename(output_p)}")
            Image.fromarray(current_image).save(output_p, 'PNG')

            self.log("\n--- Analisi completata con successo! ---")
            messagebox.showinfo("Successo", "L'analisi è stata completata con successo!")

        except FileNotFoundError:
            messagebox.showerror("Errore", f"File non trovato: {input_p}")
            self.log(f"ERRORE: File non trovato: {input_p}")
        except Exception as e:
            messagebox.showerror("Errore", f"Si è verificato un errore: {e}")
            self.log(f"ERRORE: {e}")
        finally:
            self.run_button.config(state="normal")


if __name__ == '__main__':
    app = App()
    app.mainloop()