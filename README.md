# Avian - Analizzatore di Immagini di Vetrini

**Avian** è un'applicazione desktop sviluppata in Python per l'analisi di immagini di campioni biologici su vetrini, con un focus specifico sul conteggio e la segmentazione di cellule ematiche aviarie.

L'applicazione utilizza un modello di deep learning **YOLOv8-seg** per la segmentazione delle cellule e offre un'interfaccia grafica semplice per automatizzare il processo di analisi.

## Funzionalità

- **Interfaccia Grafica Semplice**: Un'interfaccia utente intuitiva costruita con Tkinter per guidare l'utente attraverso il processo.
- **Selezione Dinamica dei File**: Permette di selezionare facilmente l'immagine di input, il percorso di output e il modello di IA da utilizzare.
- **Fase di Analisi Configurabile**:
  - **Conteggio e Segmentazione Cellule**: Utilizza un modello YOLOv8 per rilevare, classificare, contare e segmentare le cellule (es. eritrociti, trombociti, linfociti).
- **Output Multipli**:
  - **Immagine Annotata**: Salva una nuova immagine con le maschere di segmentazione e i riquadri (bounding box) disegnati attorno alle cellule rilevate, etichettati con classe e confidenza.
  - **Dati Strutturati (COCO JSON)**: Genera un file `.json` contenente le coordinate dei box, le classi e i punteggi in un formato compatibile con lo standard COCO, per eventuali analisi successive.
  - **Riepilogo Conteggi**: Mostra un riepilogo dei conteggi per ogni classe di cellule direttamente nell'interfaccia.
- **Supporto Accelerazione Hardware**: Rileva e utilizza automaticamente provider di esecuzione come CUDA (NVIDIA) o DirectML (Windows) se disponibili, altrimenti si affida alla CPU.

## Installazione

Per eseguire l'applicazione e i relativi script di addestramento, si consiglia di utilizzare un ambiente virtuale per gestire le dipendenze. Di seguito sono riportate le istruzioni per l'installazione tramite **Conda**.

1.  **Clona il repository**:
    ```bash
    git clone https://github.com/daniele-gregorio/Avian.git
    cd Avian
    ```

2.  **Installa Conda**:
    Se non hai Conda, puoi installare [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (una versione minimale di Anaconda). Scegli l'installer appropriato per il tuo sistema operativo e segui le istruzioni.

3.  **Crea e attiva l'ambiente Conda**:
    Crea un nuovo ambiente virtuale (es. chiamato `avian_env`) con una versione di Python compatibile (es. 3.9) e attivalo.
    ```bash
    conda create --name avian_env python=3.9
    conda activate avian_env
    ```

4.  **Installa le dipendenze**:
    Con l'ambiente attivo, installa i pacchetti Python necessari tramite `pip`.
    ```bash
    pip install numpy Pillow ultralytics opencv-python
    ```

5.  **Scarica il modello pre-addestrato**:
    Assicurati che il file del modello YOLOv8 (es. `yolov8_seg.pt`) sia presente nella stessa cartella dello script `main.py`, oppure selezionalo tramite l'interfaccia.

## Utilizzo

1.  **Avvia l'applicazione**:
    ```bash
    python main.py
    ```
2.  **Seleziona l'Immagine di Input** tramite il pulsante "Sfoglia...".
3.  **(Opzionale) Seleziona un Modello di IA** se vuoi usarne uno diverso da quello di default.
4.  **Scegli le Fasi di Analisi**: Di default, è attivo solo il conteggio. Il preprocessing non è necessario con i modelli YOLO.
5.  **Specifica il Percorso di Output** per l'immagine analizzata.
6.  **Clicca su "Avvia Analisi"** e attendi il completamento. I risultati e i log appariranno nella finestra principale.

## Addestramento Modello (Opzionale)

La cartella `dataset marburgo` contiene gli script necessari per addestrare un modello di segmentazione YOLOv8 personalizzato. Per procedere, segui questi passaggi nella sequenza corretta.

1.  **Prepara la struttura delle cartelle**:
    Prima di eseguire qualsiasi script, organizza il tuo dataset. All'interno della cartella `dataset marburgo`, crea una sottocartella chiamata `dataset_segmentation`. Questa cartella deve contenere le immagini e le etichette (in formato YOLO) suddivise in `train`, `val` e (opzionalmente) `test`.

    La struttura finale dovrà essere simile a questa:
    ```
    Avian/
    └── dataset marburgo/
        ├── dataset_segmentation/
        │   ├── images/
        │   │   ├── train/
        │   │   └── val/
        │   └── labels/
        │       ├── train/
        │       └── val/
        ├── prepare_dataset.py
        ├── create_yaml.py
        └── train_yolo.py
    ```

2.  **Esegui gli script di preparazione e addestramento**:
    Assicurati che il tuo ambiente Conda (`avian_env`) sia attivo. Posizionati nella cartella `dataset marburgo` ed esegui gli script in questa sequenza:
    ```bash
    cd "dataset marburgo"
    python prepare_dataset.py  # Prepara i file di configurazione del dataset
    python create_yaml.py      # Crea il file .yaml per l'addestramento di YOLO
    python train_yolo.py       # Avvia l'addestramento del modello
    ```

3.  **Trova il modello addestrato**:
    Al termine dell'addestramento, il modello migliore (`best.pt`) e altri artefatti saranno salvati in una sottocartella all'interno di `runs/segment/`, ad esempio `runs/segment/yolov8n_avian_blood_seg/weights/`. Potrai quindi utilizzare questo modello nell'applicazione principale.