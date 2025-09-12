# Avian - Analizzatore di Immagini di Vetrini

**Avian** è un'applicazione desktop sviluppata in Python per l'analisi di immagini di campioni biologici su vetrini, con un focus specifico sul conteggio e la segmentazione di cellule ematiche aviarie.

L'applicazione utilizza un modello di deep learning [YOLOv8-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) per la segmentazione delle cellule e offre un'interfaccia grafica semplice per automatizzare il processo di analisi.

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
    git clone https://github.com/danielelozzi/avian.git
    cd Avian
    ```

2.  **Installa Conda**:
    Se non hai Conda, puoi installare [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (una versione minimale di Anaconda). Scegli l'installer appropriato per il tuo sistema operativo e segui le istruzioni.

3.  **Crea e attiva l'ambiente Conda**:
    Crea un nuovo ambiente virtuale (es. chiamato `avian_env`) con una versione di Python compatibile (es. 3.9) e attivalo.
    ```bash
    conda create --name avian_env python=3.9
    conda activate avian_env
    conda install pip
    ```

4.  **Installa le dipendenze**:
    Con l'ambiente attivo, installa i pacchetti Python necessari tramite `pip`.
    ```bash
    pip install -r requirements.txt
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

La cartella del dataset, contiene gli script necessari per addestrare un modello di segmentazione YOLOv8 personalizzato. Per procedere, segui questi passaggi nella sequenza corretta.

1.  **Prepara la struttura delle cartelle**:
    Prima di eseguire qualsiasi script, organizza il tuo dataset (scaricabile da qui: [dataset marburgo](https://data.uni-marburg.de/entities/dataset/c78489de-e08c-4818-800b-1f182aa2e631)). All'interno della cartella `dataset marburgo`, crea una sottocartella chiamata `dataset_segmentation`. Questa cartella deve contenere le immagini e le etichette (in formato YOLO) suddivise in `train`, `val` e (opzionalmente) `test`.

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

3.  **Trova e utilizza il modello addestrato**:
    Al termine dell'addestramento, lo script `train_yolo.py` salverà due versioni del modello:
    -   `runs/segment/yolov8n_avian_blood_seg/weights/best.pt`: Il modello con le migliori performance sul set di validazione.
    -   `yolov8nseg_avian.pt`: Una copia del modello migliore, salvata nella cartella principale del progetto per un accesso più comodo.

    Puoi utilizzare il file `yolov8nseg_avian.pt` direttamente nell'applicazione principale, selezionandolo tramite l'interfaccia grafica.

4.  **Valuta le performance del modello (Opzionale ma consigliato)**:
    Per ottenere metriche di accuratezza quantitative (come Precision, Recall, mAP, IoU) sul set di test, esegui lo script di valutazione:
    ```bash
    # Sempre dalla cartella "dataset marburgo"
    python evaluate_model.py
    ```
    Questo script caricherà il modello `yolov8nseg_avian.pt` e calcolerà le sue performance sul set di dati di test, fornendo una valutazione oggettiva della sua efficacia nella segmentazione.

### Spiegazione delle Metriche di Valutazione

Lo script `evaluate_model.py` (e il processo di validazione durante l'addestramento) utilizza metriche standard per valutare le performance del modello di segmentazione. Ecco una breve spiegazione delle più importanti:

-   **Intersection over Union (IoU)**: È la metrica fondamentale per la segmentazione. Misura la sovrapposizione tra la maschera predetta dal modello e la maschera reale (ground truth). Viene calcolata come `(Area di Intersezione) / (Area di Unione)`. Un valore di 1.0 indica una sovrapposizione perfetta.

-   **Precision (P)**: Indica la precisione delle predizioni. Risponde alla domanda: "Di tutte le cellule che il modello ha identificato, quante erano corrette?". Un valore alto significa che il modello ha pochi falsi positivi.

-   **Recall (R)**: Indica la capacità del modello di trovare tutte le cellule rilevanti. Risponde alla domanda: "Di tutte le cellule realmente presenti nell'immagine, quante ne ha trovate il modello?". Un valore alto significa che il modello ha pochi falsi negativi (non ha "mancato" molte cellule).

-   **mAP50 (mean Average Precision @ 50%)**: È una delle metriche principali per giudicare un modello. Rappresenta la media della precisione calcolata con una soglia di IoU del 50%. In pratica, se la sovrapposizione (IoU) tra la predizione e la maschera reale è superiore al 50%, la predizione è considerata un successo. Un valore più vicino a 1.0 è migliore.

-   **mAP50-95 (mean Average Precision @ 50-95%)**: È una metrica più severa e completa. È la media delle performance (mAP) calcolata su diverse soglie di IoU, dal 50% al 95%. Un punteggio alto qui indica che il modello non solo trova le cellule, ma le segmenta anche con un contorno molto preciso.

Queste metriche vengono calcolate sia per i riquadri (`Box`) che per le maschere di segmentazione (`Mask`). Per questo progetto, le metriche relative alla **maschera** sono le più importanti.
