#!/usr/bin/env python
# coding: utf-8
 
import json
import tkinter as tk 
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import threading
import cv2
import csv
import datetime
from cell_preprocessing import preprocess_image
from cell_detector import merge_tile_results
from ultralytics.utils.plotting import Colors
from ultralytics.engine.results import Boxes, Masks


class App(tk.Tk):
    """
    Classe principale dell'applicazione GUI con Tkinter.
    """
    def __init__(self):
        super().__init__()
        self.title("Avian - Analisi Immagini Vetrini")
        self.geometry("800x600")
 
        self.model = None
        self.colors = Colors()  # Istanza per la gestione dei colori
 
        # Variabili di stato
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_path = tk.StringVar(value="yolov8nseg_avian_100epoche.pt")  # Modello di default
        self.do_enhance_contrast = tk.BooleanVar(value=False)
        self.do_isolate_and_crop = tk.BooleanVar(value=False)
        self.do_counting = tk.BooleanVar(value=True)
        self.font_size = tk.IntVar(value=12)  # Valore di default per la dimensione del font
        self.line_width = tk.IntVar(value=2)  # Valore di default per lo spessore della linea
        self.do_histogram_matching = tk.BooleanVar(value=False)

        # --- Layout ---
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding="10")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
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
        
        ttk.Checkbutton(preproc_frame, text="Isola e ritaglia campione circolare", variable=self.do_isolate_and_crop).pack(anchor="w")
        ttk.Checkbutton(preproc_frame, text="Migliora contrasto", variable=self.do_enhance_contrast).pack(anchor="w")

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

        font_frame = ttk.Frame(settings_frame)
        font_frame.pack(fill='x', pady=2)
        ttk.Label(font_frame, text="Dimensione Font Etichette:").pack(side="left", padx=(0, 5))
        ttk.Spinbox(font_frame, from_=4, to=48, increment=1, textvariable=self.font_size, width=5).pack(side="left")

        line_frame = ttk.Frame(settings_frame)
        line_frame.pack(fill='x', pady=2)
        ttk.Label(line_frame, text="Spessore Bordo Box:").pack(side="left", padx=(0, 5))
        ttk.Spinbox(line_frame, from_=1, to=20, increment=1, textvariable=self.line_width, width=5).pack(side="left")

        self.run_button = ttk.Button(scrollable_frame, text="Avvia Analisi", command=self.start_analysis_thread)
        self.run_button.pack(fill="x", pady=10)

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
        self.after(0, self.log, message)

    def _safe_messagebox(self, type, title, message):
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
            base, ext = os.path.splitext(path)
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
        self.run_button.config(state="disabled")
        self.log("--- Avvio analisi ---")
        thread = threading.Thread(target=self.run_analysis)
        thread.start()

    def _to_numpy(self, data):
        """Conversione sicura in numpy."""
        if hasattr(data, "cpu"):
            return data.cpu().numpy()
        return np.asarray(data)

    def _create_coco_output(self, results, class_names):
        if results is None or results.boxes is None or len(results.boxes) == 0:
            return {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": []}

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

        masks_xy = []
        if results.masks is not None:
            masks_np = self._to_numpy(results.masks.data)
            for mask in masks_np:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    seg = max(contours, key=cv2.contourArea).squeeze(1)
                    masks_xy.append(seg)
                else:
                    masks_xy.append(np.array([]))
        else:
            masks_xy = [[] for _ in boxes_xywh]

        for i, (box, cls_id, conf) in enumerate(zip(boxes_xywh, cls_ids, confs)):
            x_center, y_center, w, h = box
            x1 = x_center - w / 2
            y1 = y_center - h / 2

            segmentation = [masks_xy[i].flatten().tolist()] if len(masks_xy[i]) > 0 else []

            coco_annotation = {
                "id": annotation_id, "image_id": 1, "category_id": int(cls_id),
                "bbox": [float(x1), float(y1), float(w), float(h)], "area": float(w * h), "segmentation": segmentation,
                "iscrowd": 0, "score": float(conf)
            }
            coco_output["annotations"].append(coco_annotation)
            annotation_id += 1
        return coco_output

    def _draw_annotations(self, image, results, class_names, line_width, font_size):
        if not results or not results.boxes or not results.masks:
            return image

        overlay = image.copy()
        annotated_image = image.copy()
        boxes = self._to_numpy(results.boxes.data)
        
        segments = []
        if results.masks is not None and hasattr(results.masks, 'xy'):
            segments = results.masks.xy
        
        for i, (box, segment) in enumerate(zip(boxes, segments)):
            if segment.size == 0:
                continue

            class_id = int(box[5])
            score = box[4]
            color = self.colors(class_id, True)

            cv2.fillPoly(overlay, [segment.astype(np.int32)], color)

            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, line_width)

            label = f"{class_names[class_id]} {score:.2f}"
            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size / 20.0, 1)
            cv2.rectangle(annotated_image, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size / 20.0, (255, 255, 255), 1, cv2.LINE_AA)

        return cv2.addWeighted(overlay, 0.4, annotated_image, 0.6, 0)

    def run_analysis(self):
        input_p = self.input_path.get()
        output_p = self.output_path.get()
        model_p = self.model_path.get()
        try:
            if not all([input_p, output_p, model_p]):
                raise ValueError("Assicurati di aver selezionato un file di input, un modello e un percorso di output.")

            self._safe_log(f"Caricamento modello da: {os.path.basename(model_p)}")
            self.model = YOLO(model_p)
            self._safe_log(f"Modello caricato. Device: {self.model.device}")

            if not os.path.exists(input_p):
                raise FileNotFoundError(f"File di input non trovato: {input_p}")

            image_to_process = cv2.imread(input_p)
            self._safe_log(f"1. Caricamento immagine da: {os.path.basename(input_p)}")

            enhance = self.do_enhance_contrast.get()
            isolate = self.do_isolate_and_crop.get()
            do_hist_match = self.do_histogram_matching.get()

            self._safe_log(f"2. Esecuzione del preprocessing...")
            self._safe_log(f"   - Isola/Ritaglia: {isolate}, Contrasto: {enhance}")
            self._safe_log(f"   - Histogram Matching: {do_hist_match}")

            reference_image = None
            if do_hist_match and not enhance: # Non ha senso fare entrambi
                ref_path = filedialog.askopenfilename(title="Seleziona immagine di riferimento", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.tif")])
                if not ref_path:
                    raise ValueError("Histogram matching abilitato ma nessuna immagine fornita.")
                reference_image = cv2.imread(ref_path)
            
            # Chiama la funzione di preprocessing e riceve un dizionario
            preprocessing_results = preprocess_image(
                image_array=image_to_process,
                enhance_contrast=enhance, isolate_and_crop=isolate,
                do_histogram_matching=do_hist_match, reference_image=reference_image
            )

            # Estrae i dati dal dizionario
            image_for_prediction = preprocessing_results["processed_image"]
            original_processed_image = preprocessing_results["original_processed_image"]
            self._safe_log("   -> Preprocessing completato.")

            if self.do_counting.get():
                self._safe_log("3. Esecuzione rilevamento e conteggio...")
                conf_threshold, iou_threshold = 0.25, 0.45
                
                raw_results = self.model.predict(source=image_for_prediction, conf=conf_threshold, iou=iou_threshold, save=False, verbose=False)

                # Prendiamo il primo (e unico) risultato
                results = raw_results[0] if raw_results else None
                self._safe_log(f"   -> Rilevamento completato. Trovate {len(results.boxes) if results.boxes else 0} cellule.")

                if results and results.boxes is not None and len(results.boxes) > 0:
                    annotated_image_bgr = self._draw_annotations(original_processed_image, results, self.model.names, self.line_width.get(), self.font_size.get())
                    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

                    class_ids = self._to_numpy(results.boxes.cls).astype(int)
                    class_names = self.model.names
                    counts = {class_names[cid]: np.count_nonzero(class_ids == cid) for cid in np.unique(class_ids)}

                    json_path = os.path.splitext(output_p)[0] + ".json"
                    coco_data = self._create_coco_output(results, class_names)
                    coco_data['images'][0]['file_name'] = os.path.basename(output_p)
                    with open(json_path, 'w') as f:
                        json.dump(coco_data, f, indent=4)
                    self._safe_log(f"   -> Annotazioni salvate in: {os.path.basename(json_path)}")

                    csv_path = os.path.splitext(output_p)[0] + "_counts.csv"
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['classe', 'conteggio'])
                        for class_name, count in counts.items():
                            writer.writerow([class_name, count])
                    self._safe_log(f"   -> Riepilogo conteggi salvato in: {os.path.basename(csv_path)}")
                else:
                    annotated_image_rgb = cv2.cvtColor(original_processed_image, cv2.COLOR_BGR2RGB)
            else:
                annotated_image_rgb = cv2.cvtColor(original_processed_image, cv2.COLOR_BGR2RGB)

            Image.fromarray(annotated_image_rgb).save(output_p, 'PNG')
            self._safe_log(f"4. Salvataggio immagine risultato in: {os.path.basename(output_p)}")
            self._safe_messagebox("info", "Successo", "Analisi completata con successo!")

        except Exception as e:
            error_message = f"Si è verificato un errore imprevisto: {e}"
            self._safe_log(f"ERRORE: {e}")
            self._safe_messagebox("error", "Errore", error_message)
        finally:
            self.after(0, lambda: self.run_button.config(state="normal"))


if __name__ == '__main__':
    app = App()
    app.mainloop()
