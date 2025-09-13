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
import csv
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
        self.font_size = tk.IntVar(value=12) # Valore di default per la dimensione del font
        self.line_width = tk.IntVar(value=2) # Valore di default per lo spessore della linea
        self.input_magnification = tk.IntVar(value=40) # Ingrandimento immagine input
        self.train_magnification = tk.IntVar(value=100) # Ingrandimento usato in training

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

        # Sezione Impostazioni Output
        settings_frame = ttk.LabelFrame(main_frame, text="5. Impostazioni Visualizzazione", padding="10")
        settings_frame.pack(fill="x", pady=5)

        # Dimensione Font
        font_frame = ttk.Frame(settings_frame)
        font_frame.pack(fill='x', pady=2)
        ttk.Label(font_frame, text="Dimensione Font Etichette:").pack(side="left", padx=(0, 5))
        ttk.Spinbox(font_frame, from_=4, to=48, increment=1, textvariable=self.font_size, width=5).pack(side="left")

        # Spessore Bordo
        line_frame = ttk.Frame(settings_frame)
        line_frame.pack(fill='x', pady=2)
        ttk.Label(line_frame, text="Spessore Bordo Box:").pack(side="left", padx=(0, 5))
        ttk.Spinbox(line_frame, from_=1, to=20, increment=1, textvariable=self.line_width, width=5).pack(side="left")

        # Sezione Scaling
        scale_frame = ttk.LabelFrame(main_frame, text="6. Impostazioni Scala (Magnification)", padding="10")
        scale_frame.pack(fill="x", pady=5)

        # Magnification Input
        in_mag_frame = ttk.Frame(scale_frame)
        in_mag_frame.pack(fill='x', pady=2)
        ttk.Label(in_mag_frame, text="Ingrandimento Immagine Input (es. 40x):").pack(side="left", padx=(0, 5))
        ttk.Spinbox(in_mag_frame, from_=10, to=200, increment=10, textvariable=self.input_magnification, width=5).pack(side="left")

        train_mag_frame = ttk.Frame(scale_frame)
        train_mag_frame.pack(fill='x', pady=2)
        ttk.Label(train_mag_frame, text="Ingrandimento Addestramento Modello (es. 100x):").pack(side="left", padx=(0, 5))
        ttk.Spinbox(train_mag_frame, from_=10, to=200, increment=10, textvariable=self.train_magnification, width=5).pack(side="left")

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
                "bbox": [float(x1), float(y1), float(w), float(h)], "area": float(w * h),
                "segmentation": [mask.flatten().tolist()], "iscrowd": 0, "score": float(conf)
            }
            coco_output["annotations"].append(coco_annotation)
            annotation_id += 1
        return coco_output
    
    def _merge_results(self, all_results, original_image_shape):
        # Funzione di utilità per unire i risultati (per ora semplice concatenazione)
        return all_results

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
                
                # Recupera i parametri di visualizzazione dall'interfaccia
                font_size = self.font_size.get()
                line_width = self.line_width.get()

                # --- LOGICA DI SCALING E TILING ---
                input_mag = self.input_magnification.get()
                train_mag = self.train_magnification.get()
                scaling_factor = train_mag / input_mag

                original_image = cv2.imread(input_p)
                original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                
                images_to_process = []
                if scaling_factor == 1.0:
                    self.log("   -> Ingrandimenti corrispondono. Nessuno scaling necessario.")
                    images_to_process.append(original_image_rgb)
                elif scaling_factor < 1.0: # Downscaling
                    self.log(f"   -> Downscaling immagine di {scaling_factor:.2f}x...")
                    new_w = int(original_image.shape[1] * scaling_factor)
                    new_h = int(original_image.shape[0] * scaling_factor)
                    resized = cv2.resize(original_image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    images_to_process.append(resized)
                else: # Upscaling con Tiling
                    self.log(f"   -> Upscaling richiesto ({scaling_factor:.2f}x). Avvio tiling...")
                    model_input_size = 640 # Assumiamo 640, come da train_yolo.py
                    tile_size = int(model_input_size / scaling_factor)
                    overlap = int(tile_size * 0.2) # 20% di overlap
                    stride = tile_size - overlap

                    h, w, _ = original_image.shape
                    num_tiles = 0
                    for y in range(0, h, stride):
                        for x in range(0, w, stride):
                            y_end = min(y + tile_size, h)
                            x_end = min(x + tile_size, w)
                            tile = original_image_rgb[y:y_end, x:x_end]
                            
                            # Salta tile troppo piccoli
                            if tile.shape[0] < stride/2 or tile.shape[1] < stride/2:
                                continue

                            # Upscaling del tile alla dimensione attesa dal modello
                            upscaled_tile = cv2.resize(tile, (model_input_size, model_input_size), interpolation=cv2.INTER_CUBIC)
                            images_to_process.append(upscaled_tile)
                            num_tiles += 1
                    self.log(f"   -> Immagine suddivisa in {num_tiles} tasselli.")

                # Esegui predizione su tutte le immagini/tasselli preparati
                all_results = self.model.predict(source=images_to_process, conf=0.25, save=False)

                # Se abbiamo usato i tile, dobbiamo unire i risultati.
                # Per ora, disegniamo i risultati sul primo tile per visualizzazione
                # Una logica di unione completa richiederebbe di rimappare le coordinate.
                if scaling_factor > 1.0:
                    self.log("   -> ATTENZIONE: La visualizzazione mostra solo il risultato del primo tile.")
                    results = [all_results[0]] if all_results else []
                else:
                    results = all_results

                # Controlla se sono state fatte delle rilevazioni
                if results and results[0].masks is not None and len(results[0].masks) > 0:
                    # Ottieni l'immagine con le annotazioni, passando i parametri personalizzati
                    annotated_image_bgr = results[0].plot(font_size=font_size, line_width=line_width) # Disegna sul primo risultato
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

                    # Salva il file CSV con i conteggi
                    csv_path = os.path.splitext(output_p)[0] + "_counts.csv"
                    try:
                        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(['classe', 'conteggio']) # Intestazione del CSV
                            for class_name, count in counts.items():
                                writer.writerow([class_name, count])
                        self.log(f"   -> Riepilogo conteggi salvato in: {os.path.basename(csv_path)}")
                    except Exception as e:
                        self.log(f"   -> ERRORE nel salvataggio del file CSV: {e}")
                else:
                    self.log("   -> Nessuna cellula rilevata nell'immagine con la confidenza attuale.")
                    # Se non viene rilevato nulla, usa l'immagine originale
                    annotated_image_rgb = original_image_rgb

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