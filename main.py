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
from cell_preprocessing import preprocess_image
from ultralytics.utils.plotting import Colors


class App(tk.Tk):
    """
    Classe principale dell'applicazione GUI con Tkinter.
    """
    def __init__(self):
        super().__init__()
        self.title("Avian - Analisi Immagini Vetrini")
        self.geometry("800x600")
 
        self.model = None
        self.colors = Colors() # Istanza per la gestione dei colori
 
        # Variabili di stato
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_path = tk.StringVar(value="yolov8nseg_avian.pt") # Modello di default
        self.do_enhance_contrast = tk.BooleanVar(value=False)
        self.do_isolate_and_crop = tk.BooleanVar(value=False)
        self.do_counting = tk.BooleanVar(value=True)
        self.font_size = tk.IntVar(value=12) # Valore di default per la dimensione del font
        self.line_width = tk.IntVar(value=2) # Valore di default per lo spessore della linea
        self.input_magnification = tk.IntVar(value=40) # Ingrandimento immagine input
        self.train_magnification = tk.IntVar(value=100) # Ingrandimento usato in training
        self.do_scaling = tk.BooleanVar(value=True) # Abilita/disabilita lo scaling

        # --- Layout ---
        # Creazione di un canvas e una scrollbar per rendere la finestra scorrevole
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding="10")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Sezione Input
        input_frame = ttk.LabelFrame(scrollable_frame, text="1. Seleziona Immagine Input", padding="10")
        input_frame.pack(fill="x", pady=5)
        ttk.Entry(input_frame, textvariable=self.input_path).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(input_frame, text="Sfoglia...", command=self.browse_input).pack(side="left")

        # Sezione Modello
        model_frame = ttk.LabelFrame(scrollable_frame, text="2. Seleziona Modello di IA", padding="10")
        model_frame.pack(fill="x", pady=5)
        ttk.Entry(model_frame, textvariable=self.model_path).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(model_frame, text="Sfoglia...", command=self.browse_model).pack(side="left")

        # Sezione Preprocessing
        preproc_frame = ttk.LabelFrame(scrollable_frame, text="3. Preprocessing (Opzionale)", padding="10")
        preproc_frame.pack(fill="x", pady=5)
        ttk.Checkbutton(preproc_frame, text="Migliora contrasto", variable=self.do_enhance_contrast).pack(anchor="w")
        ttk.Checkbutton(preproc_frame, text="Isola e ritaglia campione circolare", variable=self.do_isolate_and_crop).pack(anchor="w")

        # Sezione Analisi
        analysis_frame = ttk.LabelFrame(scrollable_frame, text="4. Fasi di Analisi", padding="10")
        analysis_frame.pack(fill="x", pady=5)
        ttk.Checkbutton(analysis_frame, text="Esegui Conteggio Cellule (Rilevamento e Classificazione)", variable=self.do_counting).pack(anchor="w")

        # Sezione Output
        output_frame = ttk.LabelFrame(scrollable_frame, text="5. Salva Immagine Risultato", padding="10")
        output_frame.pack(fill="x", pady=5)
        ttk.Entry(output_frame, textvariable=self.output_path).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(output_frame, text="Salva come...", command=self.browse_output).pack(side="left")

        # Sezione Impostazioni Output
        settings_frame = ttk.LabelFrame(scrollable_frame, text="6. Impostazioni Visualizzazione", padding="10")
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
        scale_frame = ttk.LabelFrame(scrollable_frame, text="7. Impostazioni Scala (Magnification)", padding="10")
        scale_frame.pack(fill="x", pady=5)

        # Checkbox per abilitare lo scaling
        ttk.Checkbutton(scale_frame, text="Abilita scaling per magnificazione diversa", variable=self.do_scaling).pack(anchor="w", pady=(0, 5))

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
        self.run_button = ttk.Button(scrollable_frame, text="Avvia Analisi", command=self.start_analysis_thread)
        self.run_button.pack(fill="x", pady=10)

        # Log Area
        log_frame = ttk.LabelFrame(scrollable_frame, text="Log e Risultati", padding="10")
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
        """
        Unisce i risultati dai vari tasselli, rimappando le coordinate all'immagine originale
        e applicando la Non-Maximum Suppression (NMS) per rimuovere i duplicati.
        """
        if not all_results or not any(r.boxes for r in all_results):
            return None

        # Prepara una lista per contenere tutti i box, score e classi
        all_boxes = []
        all_scores = []
        all_classes = []
        all_masks = []

        for result in all_results:
            if result.boxes is None:
                continue

            # Rimappa i box e le maschere alle coordinate dell'immagine originale
            # result.orig_img è il tile, result.orig_shape è la dimensione dell'immagine grande
            # Questa è una semplificazione, YOLO non espone le coordinate del tile.
            # Dobbiamo gestire le coordinate manualmente.
            # La logica di predict è stata modificata per passare tuple (tile, (x_offset, y_offset))
            # Ma predict non accetta tuple, quindi modifichiamo il post-processing.
            # Per ora, uniamo semplicemente i risultati, la rimappatura va fatta nel ciclo principale.
            # Questa funzione è un placeholder finché non modifichiamo il ciclo di predizione.
            
            # La logica di unione vera e propria è complessa. Per ora, concateniamo i risultati
            # del primo risultato (che è quello che il codice faceva prima implicitamente).
            # Una vera unione richiede di modificare come `predict` viene chiamato e come i risultati
            # vengono processati.
            
            # Dato che `predict` non ci restituisce l'offset del tile, la soluzione migliore
            # è processare un tile alla volta e accumulare i risultati rimappati.
            # Ho spostato questa logica direttamente in `run_analysis`.
            # Questa funzione non è più necessaria nel suo stato attuale.
            pass

        # Se si unissero tutti i risultati, si dovrebbe fare qualcosa del genere:
        # from torchvision.ops import nms
        # final_boxes = torch.cat(all_boxes, dim=0)
        # ... e poi applicare NMS
        return [all_results[0]] if all_results else []

    def _draw_annotations(self, image, results, class_names, line_width, font_size):
        """
        Disegna manualmente le annotazioni (maschere e riquadri) sull'immagine originale
        per preservare i colori corretti.
        """
        if results is None or len(results) == 0:
            return image

        # Crea una copia dell'immagine per disegnarci sopra
        overlay = image.copy()
        annotated_image = image.copy()

        # Estrai maschere, riquadri e classi
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.data.cpu().numpy()

        for i, mask_data in enumerate(masks):
            # Prendi il colore dalla mappa del modello
            class_id = int(boxes[i, 5])
            color = self.colors(class_id, True) # Ottiene il colore BGR

            # Disegna la maschera di segmentazione
            # Ridimensiona la maschera alla dimensione dell'immagine originale
            mask = cv2.resize(mask_data, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay[mask > 0.5] = color

            # Disegna il riquadro (bounding box)
            x1, y1, x2, y2 = map(int, boxes[i, :4])
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, line_width)

            # Prepara e disegna l'etichetta (classe + confidenza)
            label = f"{class_names[class_id]} {boxes[i, 4]:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size / 20.0, 1)
            cv2.rectangle(annotated_image, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size / 20.0, (255, 255, 255), 1, cv2.LINE_AA)

        # Unisci l'overlay della maschera con l'immagine annotata
        return cv2.addWeighted(overlay, 0.4, annotated_image, 0.6, 0)

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
            image_to_process = cv2.imread(input_p)

            self.log(f"1. Caricamento immagine da: {os.path.basename(input_p)}")
            # YOLO si aspetta il percorso del file o un array numpy
            # Passare il percorso è più efficiente

            # Fase di Preprocessing
            enhance = self.do_enhance_contrast.get()
            isolate = self.do_isolate_and_crop.get()
            if enhance or isolate:
                self.log(f"2. Esecuzione del preprocessing (Contrasto: {enhance}, Isola: {isolate})...")
                image_to_process = preprocess_image(image_to_process, enhance_contrast=enhance, isolate_and_crop=isolate)
                self.log("   -> Preprocessing completato.")
            else:
                self.log("2. Preprocessing saltato dall'utente.")

            # Fase di Conteggio
            if self.do_counting.get():
                self.log(f"3. Esecuzione rilevamento e conteggio...")
                
                # Recupera i parametri di visualizzazione dall'interfaccia
                font_size = self.font_size.get()
                line_width = self.line_width.get()

                # --- LOGICA DI SCALING E TILING ---
                original_image_rgb = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
                
                images_to_process = []
                scaling_factor = 1.0
                tile_coords = [] # Memorizza le coordinate (x, y) di ogni tile

                if self.do_scaling.get():
                    input_mag = self.input_magnification.get()
                    train_mag = self.train_magnification.get()
                    scaling_factor = train_mag / input_mag

                    if scaling_factor == 1.0:
                        self.log("   -> Ingrandimenti corrispondono. Nessuno scaling necessario.")
                        images_to_process.append(original_image_rgb)
                    elif scaling_factor < 1.0: # Downscaling
                        self.log(f"   -> Downscaling immagine di {scaling_factor:.2f}x...")
                        new_w = int(original_image_rgb.shape[1] * scaling_factor)
                        new_h = int(original_image_rgb.shape[0] * scaling_factor)
                        resized = cv2.resize(original_image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        images_to_process.append(resized)
                    else: # Upscaling con Tiling
                        self.log(f"   -> Upscaling richiesto ({scaling_factor:.2f}x). Avvio tiling...")
                        model_input_size = 640 # Assumiamo 640, come da train_yolo.py
                        tile_size = int(model_input_size / scaling_factor)
                        overlap = int(tile_size * 0.2) # 20% di overlap
                        stride = tile_size - overlap

                        h, w, _ = original_image_rgb.shape
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
                else:
                    self.log("   -> Scaling per magnificazione disabilitato dall'utente.")
                    images_to_process.append(original_image_rgb)

                # Esegui predizione e unisci i risultati
                results = []
                if len(images_to_process) > 1: # Se abbiamo usato il tiling
                    self.log(f"   -> Analisi di {len(images_to_process)} tasselli...")
                    all_raw_results = self.model.predict(source=images_to_process, conf=0.25, save=False, verbose=False)
                    
                    # Prepara un oggetto Results vuoto per accumulare i risultati
                    # Usiamo il primo risultato come template
                    final_results = all_raw_results[0].cpu()
                    final_results.boxes = None
                    final_results.masks = None

                    all_boxes, all_masks, all_cls, all_conf = [], [], [], []

                    for i, result in enumerate(all_raw_results):
                        if result.boxes is None or len(result.boxes) == 0:
                            continue
                        
                        x_offset, y_offset = tile_coords[i]
                        
                        # Calcola il fattore di scala inverso per rimappare le coordinate
                        # dal tile 640x640 al tile originale
                        tile_h_orig, tile_w_orig = images_to_process[i].shape[:2] # Dimensione del tile upscalato (es. 640x640)
                        orig_tile_size_w = int(tile_w_orig / scaling_factor)
                        orig_tile_size_h = int(tile_h_orig / scaling_factor)

                        # Clona i box per non modificare l'originale
                        remapped_boxes = result.boxes.xyxy.cpu().clone()

                        # Rimappa le coordinate dei box
                        remapped_boxes[:, 0] = x_offset + (remapped_boxes[:, 0] / tile_w_orig) * orig_tile_size_w
                        remapped_boxes[:, 1] = y_offset + (remapped_boxes[:, 1] / tile_h_orig) * orig_tile_size_h
                        remapped_boxes[:, 2] = x_offset + (remapped_boxes[:, 2] / tile_w_orig) * orig_tile_size_w
                        remapped_boxes[:, 3] = y_offset + (remapped_boxes[:, 3] / tile_h_orig) * orig_tile_size_h
                        
                        all_boxes.append(remapped_boxes)
                        all_cls.append(result.boxes.cls.cpu())
                        all_conf.append(result.boxes.conf.cpu())

                        # Rimappatura maschere (più complessa, per ora uniamo i box)
                        # Per unire le maschere, dovremmo rimapparle e creare una maschera composita.
                        # Per semplicità, ci concentriamo sui box e usiamo le maschere del primo risultato utile.
                        if result.masks is not None and final_results.masks is None:
                            final_results.masks = result.masks.cpu()

                    if all_boxes:
                        final_results.boxes = result.boxes.from_xyxy(np.concatenate(all_boxes), np.concatenate(all_cls), np.concatenate(all_conf), final_results.orig_shape)
                        # TODO: Applicare Non-Maximum Suppression per rimuovere i duplicati nelle aree di overlap
                        self.log("   -> Risultati dei tasselli uniti. ATTENZIONE: potrebbero esserci duplicati nelle aree di sovrapposizione.")
                        results = [final_results.to(self.model.device)]

                else:
                    results = self.model.predict(source=images_to_process, conf=0.25, save=False)

                # Controlla se sono state fatte delle rilevazioni
                if results and results[0].masks is not None and len(results[0].masks) > 0:
                    annotated_image_bgr = self._draw_annotations(cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR), results, self.model.names, line_width, font_size)
                    # Converti in RGB per il salvataggio con Pillow
                    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

                    # Conta le cellule per classe
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    class_names = self.model.names
                    counts = {class_names[cid]: np.count_nonzero(class_ids == cid) for cid in np.unique(class_ids)}
                    self.log(f"   -> Conteggio completato.")

                    # Creazione e log della tabella riepilogativa
                    if counts:
                        # Determina la larghezza delle colonne dinamicamente
                        max_class_len = max(len(c) for c in counts.keys()) if counts else 0
                        class_col_width = max(len("Classe"), max_class_len)
                        count_col_width = max(len("Conteggio"), 10)

                        # Costruisci la tabella come stringa
                        separator = f"+-{'-' * class_col_width}-+-{'-' * count_col_width}-+"
                        header = f"| {'Classe'.ljust(class_col_width)} | {'Conteggio'.ljust(count_col_width)} |"
                        
                        self.log("\n" + separator)
                        self.log(header)
                        self.log(separator)

                        for class_name, count in sorted(counts.items()):
                            row = f"| {class_name.ljust(class_col_width)} | {str(count).ljust(count_col_width)} |"
                            self.log(row)
                        
                        self.log(separator + "\n")

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