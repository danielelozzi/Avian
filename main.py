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
from cell_preprocessing import preprocess_image, manual_histogram_matching, tile_image
from cell_detector import merge_tile_results
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
        self.input_magnification = tk.IntVar(value=100) # Ingrandimento immagine input
        self.train_magnification = tk.IntVar(value=100) # Ingrandimento usato in training
        self.do_scaling = tk.BooleanVar(value=True) # Abilita/disabilita lo scaling
        self.do_histogram_matching = tk.BooleanVar(value=False)
        self.do_tiling = tk.BooleanVar(value=False)

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
        
        # Ordine corretto come da richiesta
        ttk.Checkbutton(preproc_frame, text="Isola e ritaglia campione circolare", variable=self.do_isolate_and_crop).pack(anchor="w")
        ttk.Checkbutton(preproc_frame, text="Migliora contrasto", variable=self.do_enhance_contrast).pack(anchor="w")
        ttk.Checkbutton(preproc_frame, text="Suddividi in Tasselli (Tiling)", variable=self.do_tiling).pack(anchor="w")

        # --- Scaling (spostato qui) ---
        ttk.Checkbutton(preproc_frame, text="Abilita scaling per magnificazione diversa", variable=self.do_scaling).pack(anchor="w", pady=(5, 0))
        
        in_mag_frame = ttk.Frame(preproc_frame)
        in_mag_frame.pack(fill='x', pady=2, padx=(20, 0)) # Indentato per chiarezza
        ttk.Label(in_mag_frame, text="Ingrandimento Immagine Input (es. 40x):").pack(side="left", padx=(0, 5))
        ttk.Spinbox(in_mag_frame, from_=10, to=200, increment=10, textvariable=self.input_magnification, width=5).pack(side="left")

        train_mag_frame = ttk.Frame(preproc_frame)
        train_mag_frame.pack(fill='x', pady=2, padx=(20, 0)) # Indentato per chiarezza
        ttk.Label(train_mag_frame, text="Ingrandimento Addestramento Modello (es. 100x):").pack(side="left", padx=(0, 5))
        ttk.Spinbox(train_mag_frame, from_=10, to=200, increment=10, textvariable=self.train_magnification, width=5).pack(side="left")

        # --- Histogram Matching (ultimo) ---
        ttk.Checkbutton(preproc_frame, text="Histogram Matching (richiede immagine di riferimento)", variable=self.do_histogram_matching).pack(anchor="w")


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

    def _safe_log(self, message):
        """Logga un messaggio in modo thread-safe."""
        self.after(0, self.log, message)

    def _safe_messagebox(self, type, title, message):
        """Mostra una messagebox in modo thread-safe."""
        if type == "info":
            self.after(0, lambda: messagebox.showinfo(title, message))
        elif type == "error":
            self.after(0, lambda: messagebox.showerror(title, message))
        elif type == "warning":
            self.after(0, lambda: messagebox.showwarning(title, message))

    def browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.tif")])
        if path:
            self.input_path.set(path)
            # Propone un nome di output di default
            base, ext = os.path.splitext(path)
            # Aggiunge un timestamp per evitare sovrascritture accidentali
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path.set(f"{base}_analizzato_{timestamp}.png")

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
        self.log("--- Avvio analisi ---") # Questa è sicura perché chiamata dal thread principale
        thread = threading.Thread(target=self.run_analysis)
        thread.start()

    def _to_numpy(self, data):
        """Converte i dati in un array NumPy se sono tensori PyTorch."""
        if hasattr(data, 'cpu'):  # Controlla se è un tensore
            return data.cpu().numpy()
        return np.asarray(data) # Altrimenti, è già un array o simile

    def _create_coco_output(self, results, class_names):
        # Se non ci sono risultati, restituisci una struttura COCO vuota.
        if results is None or results.boxes is None or len(results.boxes) == 0:
            return {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": []}

        # Gestisce sia shape a 2 che a 3 dimensioni
        original_h, original_w = results.orig_shape[:2]
        coco_output = {
            "info": {"description": "Avian Cell Annotations (YOLOv8)", "version": "1.0", "date_created": datetime.date.today().isoformat()},
            "licenses": [{"id": 1, "name": "N/A", "url": ""}],
            "images": [{"id": 1, "width": original_w, "height": original_h, "file_name": "image.png", "license": 1}],
            "annotations": [],
            "categories": [{"id": i, "name": name, "supercategory": "cell"} for i, name in class_names.items()]
        }
        
        annotation_id = 1
        boxes_xywh = self._to_numpy(results.boxes.xywh)
        cls_ids = self._to_numpy(results.boxes.cls)
        confs = self._to_numpy(results.boxes.conf)

        # Controlla se le maschere sono presenti
        has_masks = results.masks is not None
        masks_xy = results.masks.xy if has_masks else [[] for _ in boxes_xywh]

        for i, (box, cls_id, conf) in enumerate(zip(boxes_xywh, cls_ids, confs)):
            x_center, y_center, w, h = box
            x1 = x_center - w / 2
            y1 = y_center - h / 2

            segmentation = [masks_xy[i].flatten().tolist()] if has_masks and len(masks_xy[i]) > 0 else []

            coco_annotation = {
                "id": annotation_id, "image_id": 1, "category_id": int(cls_id),
                "bbox": [float(x1), float(y1), float(w), float(h)], "area": float(w * h), "segmentation": segmentation,
                "iscrowd": 0, "score": float(conf)
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
        if not results or not results[0] or results[0].masks is None:
            return image

        # Crea una copia dell'immagine per disegnarci sopra
        overlay = image.copy()
        annotated_image = image.copy()

        # Estrai maschere, riquadri e classi
        masks = self._to_numpy(results[0].masks.data)
        boxes = self._to_numpy(results[0].boxes.data)

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
            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size / 20.0, 1)
            cv2.rectangle(annotated_image, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size / 20.0, (255, 255, 255), 1, cv2.LINE_AA)

        # Unisci l'overlay della maschera con l'immagine annotata
        return cv2.addWeighted(overlay, 0.4, annotated_image, 0.6, 0)

    def run_analysis(self):
        input_p = self.input_path.get()
        output_p = self.output_path.get()
        model_p = self.model_path.get()

        try:
            if not all([input_p, output_p, model_p]):
                raise ValueError("Assicurati di aver selezionato un file di input, un modello e un percorso di output.")

            self._safe_log(f"Caricamento modello da: {os.path.basename(model_p)}")
            try:
                self.model = YOLO(model_p)
                self._safe_log(f"Modello caricato. Device: {self.model.device}")
            except Exception as e:
                self._safe_log(f"ERRORE: Impossibile caricare il modello YOLO.\n{e}")
                self._safe_messagebox("error", "Errore Modello", f"Impossibile caricare il modello.\n\nDettagli: {e}")
                raise

            if not os.path.exists(input_p):
                raise FileNotFoundError(f"File di input non trovato: {input_p}")

            image_to_process = cv2.imread(input_p)

            self._safe_log(f"1. Caricamento immagine da: {os.path.basename(input_p)}")

            # --- Fase di Preprocessing ---
            enhance = self.do_enhance_contrast.get()
            isolate = self.do_isolate_and_crop.get()
            do_hist_match = self.do_histogram_matching.get()
            do_tile = self.do_tiling.get()
            do_scale = self.do_scaling.get()
            in_mag = self.input_magnification.get()
            train_mag = self.train_magnification.get()

            self._safe_log(f"2. Esecuzione del preprocessing...")
            self._safe_log(f"   - Isola/Ritaglia: {isolate}, Contrasto: {enhance}, Tiling: {do_tile}")
            self._safe_log(f"   - Scaling: {do_scale} (da {in_mag}x a {train_mag}x)")
            self._safe_log(f"   - Histogram Matching: {do_hist_match}")

            reference_image = None
            if do_hist_match:
                ref_path = filedialog.askopenfilename(title="Seleziona l'immagine di riferimento per l'istogramma", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.tif")])
                if not ref_path:
                    self._safe_log("   -> ERRORE: Nessuna immagine di riferimento selezionata. Analisi interrotta.")
                    raise ValueError("Histogram matching abilitato ma immagine di riferimento non fornita.")
                self._safe_log(f"   -> Caricamento immagine di riferimento: {os.path.basename(ref_path)}")
                reference_image = cv2.imread(ref_path)

            # Esegui il preprocessing
            model_input_size = 640
            overlap = int(model_input_size * 0.2)
            
            images_for_prediction, coords, original_processed_image = preprocess_image(
                image_to_process,
                enhance_contrast=enhance,
                isolate_and_crop=isolate,
                do_histogram_matching=do_hist_match,
                reference_image=reference_image,
                do_tiling=do_tile,
                tile_size=model_input_size,
                overlap=overlap
            )
            self._safe_log("   -> Preprocessing completato.")

            # --- Fase di Conteggio ---
            if self.do_counting.get():
                self._safe_log(f"3. Esecuzione rilevamento e conteggio...")
                results = None

                # Esegui la predizione sulle immagini (singola o tasselli)
                self._safe_log(f"   -> Esecuzione del modello su {len(images_for_prediction)} immagini...")
                raw_results = self.model.predict(source=images_for_prediction, conf=0.25, save=False, verbose=False)

                if do_tile:
                    # Unisci i risultati se è stato fatto il tiling
                    self._safe_log("   -> Unione dei risultati dei tasselli e applicazione di NMS...")
                    results_with_coords = list(zip(raw_results, coords))
                    results = merge_tile_results(results_with_coords, original_shape=original_processed_image.shape, conf_threshold=0.25, iou_threshold=0.45)
                    self._safe_log("   -> Unione completata.")
                else:
                    # Se non c'è tiling, prendi il primo (e unico) risultato
                    results = raw_results[0] if raw_results else None

                # Controlla se sono state fatte delle rilevazioni
                if results and results.boxes is not None and len(results.boxes) > 0:
                    # Per il disegno, passiamo un solo oggetto Results in una lista
                    # La funzione _draw_annotations si aspetta una lista
                    font_size = self.font_size.get()
                    line_width = self.line_width.get()
                    annotated_image_bgr = self._draw_annotations(original_processed_image, [results], self.model.names, line_width, font_size)
                    # Converti in RGB per il salvataggio con Pillow
                    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

                    # Conta le cellule per classe
                    class_ids = self._to_numpy(results.boxes.cls).astype(int)
                    class_names = self.model.names
                    counts = {class_names[cid]: np.count_nonzero(class_ids == cid) for cid in np.unique(class_ids)}
                    self._safe_log(f"   -> Conteggio completato.")

                    # Creazione e log della tabella riepilogativa
                    if counts:
                        # Determina la larghezza delle colonne dinamicamente
                        max_class_len = max(len(c) for c in counts.keys()) if counts else 0
                        class_col_width = max(len("Classe"), max_class_len)
                        count_col_width = max(len("Conteggio"), 10)

                        # Costruisci la tabella come stringa
                        separator = f"+-{'-' * class_col_width}-+-{'-' * count_col_width}-+"
                        header = f"| {'Classe'.ljust(class_col_width)} | {'Conteggio'.ljust(count_col_width)} |"
                        
                        self._safe_log("\n" + separator)
                        self._safe_log(header)
                        self._safe_log(separator)

                        for class_name, count in sorted(counts.items()):
                            row = f"| {class_name.ljust(class_col_width)} | {str(count).ljust(count_col_width)} |"
                            self._safe_log(row)
                        
                        self._safe_log(separator + "\n")

                    # Salva il file JSON
                    json_path = os.path.splitext(output_p)[0] + ".json"
                    coco_data = self._create_coco_output(results, class_names)
                    coco_data['images'][0]['file_name'] = os.path.basename(output_p)

                    with open(json_path, 'w') as f:
                        json.dump(coco_data, f, indent=4)
                    self._safe_log(f"   -> Annotazioni salvate in: {os.path.basename(json_path)}")

                    # Salva il file CSV con i conteggi
                    csv_path = os.path.splitext(output_p)[0] + "_counts.csv"
                    try:
                        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(['classe', 'conteggio']) # Intestazione del CSV
                            for class_name, count in counts.items():
                                writer.writerow([class_name, count])
                        self._safe_log(f"   -> Riepilogo conteggi salvato in: {os.path.basename(csv_path)}")
                    except Exception as e:
                        self._safe_log(f"   -> ERRORE nel salvataggio del file CSV: {e}")
                else:
                    self._safe_log("   -> Nessuna cellula rilevata nell'immagine con la confidenza attuale.")
                    # Se non viene rilevato nulla, usa l'immagine processata (o originale)
                    annotated_image_rgb = cv2.cvtColor(original_processed_image, cv2.COLOR_BGR2RGB)

            else:
                self._safe_log("3. Conteggio cellule saltato.")
                # Se il conteggio è saltato, usa l'immagine processata
                annotated_image_rgb = cv2.cvtColor(original_processed_image, cv2.COLOR_BGR2RGB)

            # Salvataggio
            self._safe_log(f"4. Salvataggio immagine risultato in: {os.path.basename(output_p)}")
            Image.fromarray(annotated_image_rgb).save(output_p, 'PNG')

            self._safe_log("\n--- Analisi completata con successo! ---")
            self._safe_messagebox("info", "Successo", "L'analisi è stata completata con successo!")

        except Exception as e:
            error_message = f"Si è verificato un errore imprevisto: {e}"
            self._safe_log(f"ERRORE: {e}")
            self._safe_messagebox("error", "Errore", error_message)
        finally:
            # Riabilita il pulsante in modo thread-safe
            self.after(0, lambda: self.run_button.config(state="normal"))


if __name__ == '__main__':
    app = App()
    app.mainloop()