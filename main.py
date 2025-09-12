#!/usr/bin/env python
# coding: utf-8

import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import threading
import cv2
import datetime


class App(tk.Tk):
    """
    Classe principale dell'applicazione GUI con Tkinter.
    """
    def __init__(self):
        super().__init__()
        self.title("Avian - Analisi Immagini Vetrini")
        self.geometry("800x600")
 
        self.model = None
 
        # Variabili di stato
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_path = tk.StringVar(value="yolov8_seg.pt") # Modello di default
        self.do_preprocessing = tk.BooleanVar(value=False) # YOLO fa già il suo preprocessing
        self.do_counting = tk.BooleanVar(value=True)

        # --- Layout ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Sezione Input
        input_frame = ttk.LabelFrame(main_frame, text="1. Seleziona Immagine Input", padding="10")
        input_frame.pack(fill="x", pady=5)
        ttk.Entry(input_frame, textvariable=self.input_path).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(input_frame, text="Sfoglia...", command=self.browse_input).pack(side="left")

        # Sezione Modello
        model_frame = ttk.LabelFrame(main_frame, text="2. Seleziona Modello di IA", padding="10")
        model_frame.pack(fill="x", pady=5)
        ttk.Entry(model_frame, textvariable=self.model_path).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(model_frame, text="Sfoglia...", command=self.browse_model).pack(side="left")

        # Sezione Fasi di Analisi
        steps_frame = ttk.LabelFrame(main_frame, text="3. Seleziona Fasi di Analisi", padding="10")
        steps_frame.pack(fill="x", pady=5)

        self.preproc_check = ttk.Checkbutton(steps_frame, text="Esegui Preprocessing (Opzionale, non richiesto da YOLO)", variable=self.do_preprocessing)
        self.preproc_check.pack(anchor="w")

        # Checkbox per il conteggio
        ttk.Checkbutton(steps_frame, text="Esegui Conteggio Cellule (Rilevamento e Classificazione)", variable=self.do_counting).pack(anchor="w", pady=(10, 0))

        # Sezione Output
        output_frame = ttk.LabelFrame(main_frame, text="4. Salva Immagine Risultato", padding="10")
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
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.tif")])
        if path:
            self.input_path.set(path)
            # Propone un nome di output di default
            base, ext = os.path.splitext(path)
            self.output_path.set(f"{base}_analizzato.png")

    def browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if path:
            self.output_path.set(path)

    def browse_model(self):
        path = filedialog.askopenfilename(title="Seleziona il modello YOLOv8", filetypes=[("PyTorch Model", "*.pt")])
        if path:
            self.model_path.set(path)

    def start_analysis_thread(self):
        # Esegue l'analisi in un thread separato per non bloccare la GUI
        self.run_button.config(state="disabled")
        self.log("--- Avvio analisi ---")
        thread = threading.Thread(target=self.run_analysis)
        thread.start()

    def _create_coco_output(self, results, class_names):
        original_h, original_w = results.orig_shape
        coco_output = {
            "info": {"description": "Avian Cell Annotations (YOLOv8)", "version": "1.0", "date_created": datetime.date.today().isoformat()},
            "licenses": [{"id": 1, "name": "N/A", "url": ""}],
            "images": [{"id": 1, "width": original_w, "height": original_h, "file_name": "image.png", "license": 1}],
            "annotations": [],
            "categories": [{"id": i, "name": name, "supercategory": "cell"} for i, name in class_names.items()]
        }
        
        annotation_id = 1
        for box, mask, cls_id, conf in zip(results.boxes.xywh.cpu().numpy(), results.masks.xy, results.boxes.cls.cpu().numpy(), results.boxes.conf.cpu().numpy()):
            x_center, y_center, w, h = box
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            
            coco_annotation = {
                "id": annotation_id, "image_id": 1, "category_id": int(cls_id),
                "bbox": [x1, y1, w, h], "area": w * h,
                "segmentation": [mask.flatten().tolist()], "iscrowd": 0, "score": float(conf)
            }
            coco_output["annotations"].append(coco_annotation)
            annotation_id += 1
        return coco_output

    def run_analysis(self):
        input_p = self.input_path.get()
        output_p = self.output_path.get()
        model_p = self.model_path.get()

        if not all([input_p, output_p, model_p]):
            messagebox.showerror("Errore", "Assicurati di aver selezionato un file di input, un modello e un percorso di output.")
            self.run_button.config(state="normal")
            return

        try:
            self.log(f"Caricamento modello da: {os.path.basename(model_p)}")
            try:
                self.model = YOLO(model_p)
                self.log(f"Modello caricato. Device: {self.model.device}")
            except Exception as e:
                self.log(f"ERRORE: Impossibile caricare il modello YOLO.\n{e}")
                messagebox.showerror("Errore Modello", f"Impossibile caricare il modello.\n\nDettagli: {e}")
                raise

            self.log(f"1. Caricamento immagine da: {os.path.basename(input_p)}")
            # YOLO si aspetta il percorso del file o un array numpy
            # Passare il percorso è più efficiente

            # Fase di Preprocessing
            if self.do_preprocessing.get():
                self.log("2. Esecuzione del preprocessing (non implementato per YOLO)...")
                # Qui potrebbe essere inserito un preprocessing custom se necessario
                # Per ora, lo saltiamo dato che YOLO lo gestisce internamente.
            else:
                self.log("2. Preprocessing saltato.")

            # Fase di Conteggio
            if self.do_counting.get():
                self.log(f"3. Esecuzione rilevamento e conteggio...")
                results = self.model.predict(source=input_p, conf=0.25, save=False)
                
                # Ottieni l'immagine con le annotazioni
                annotated_image_bgr = results[0].plot() # Restituisce un array BGR
                annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

                # Conta le cellule per classe
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                class_names = self.model.names
                counts = {class_names[cid]: np.count_nonzero(class_ids == cid) for cid in np.unique(class_ids)}
                self.log(f"   -> Conteggio completato. Risultati: {counts}")

                # Salva il file JSON
                json_path = os.path.splitext(output_p)[0] + ".json"
                coco_data = self._create_coco_output(results[0], class_names)
                coco_data['images'][0]['file_name'] = os.path.basename(output_p)

                with open(json_path, 'w') as f:
                    json.dump(coco_data, f, indent=4)
                self.log(f"   -> Annotazioni salvate in: {os.path.basename(json_path)}")
            else:
                self.log("3. Conteggio cellule saltato.")
                # Se il conteggio è saltato, carica l'immagine originale per il salvataggio
                annotated_image_rgb = np.array(Image.open(input_p))

            # Salvataggio
            self.log(f"4. Salvataggio immagine risultato in: {os.path.basename(output_p)}")
            Image.fromarray(annotated_image_rgb).save(output_p, 'PNG')

            self.log("\n--- Analisi completata con successo! ---")
            messagebox.showinfo("Successo", "L'analisi è stata completata con successo!")

        except FileNotFoundError:
            messagebox.showerror("Errore", f"File non trovato: {input_p}")
            self.log(f"ERRORE: File non trovato: {input_p}")
        except Exception as e:
            messagebox.showerror("Errore", f"Si è verificato un errore: {e}")
            # L'errore specifico viene già loggato dove si verifica
        finally:
            self.run_button.config(state="normal")


if __name__ == '__main__':
    app = App()
    app.mainloop()