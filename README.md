# Avian - Analizzatore di Immagini di Vetrini Ematologici

**Avian** è un'applicazione desktop sviluppata in Python per l'analisi di immagini di campioni biologici su vetrini, con un focus specifico sul conteggio e la segmentazione di cellule ematiche aviarie.

L'applicazione sfrutta un modello di deep learning basato su **YOLOv8-seg** per la segmentazione di istanza e offre un'interfaccia grafica intuitiva per automatizzare l'intero processo di analisi, dal preprocessing dell'immagine all'esportazione dei risultati.

## 🚀 Funzionalità Principali

- **Interfaccia Grafica Semplice**: Un'interfaccia utente intuitiva costruita con Tkinter per guidare l'utente attraverso il processo.
- **Selezione Dinamica dei File**: Permette di selezionare facilmente l'immagine di input, il percorso di output e il modello di IA da utilizzare.
- **Preprocessing Avanzato (Opzionale)**:
  - **Isolamento del Campione**: Isola e ritaglia automaticamente il campione circolare dal resto del vetrino.
  - **Miglioramento del Contrasto**: Applica l'equalizzazione adattiva dell'istogramma (CLAHE) per migliorare la visibilità delle cellule.
  - **Normalizzazione del Colore**: Utilizza l'Histogram Matching per standardizzare l'aspetto delle immagini rispetto a un'immagine di riferimento, riducendo la variabilità dovuta alla colorazione.
- **Analisi basata su Deep Learning**:
  - **Rilevamento, Classificazione e Segmentazione**: Utilizza un modello YOLOv8 per rilevare, classificare e segmentare con precisione ogni singola cellula (es. eritrociti, trombociti, linfociti).
- **Esportazione Completa dei Risultati**:
  - **Immagine Annotata**: Salva una nuova immagine con le maschere di segmentazione e i riquadri (bounding box) disegnati attorno alle cellule rilevate, etichettati con classe e confidenza.
  - **Dati Strutturati (COCO JSON)**: Genera un file `.json` contenente le annotazioni (box, maschere, classi, score) in un formato compatibile con lo standard COCO, ideale per ulteriori analisi o per l'integrazione con altri tool.
  - **Riepilogo Conteggi (CSV)**: Crea un file `.csv` con il conteggio totale per ogni classe di cellule identificata.
- **Personalizzazione dell'Output**:
  - Permette di regolare lo spessore delle linee e la dimensione del font delle etichette sull'immagine annotata.
- **Supporto Accelerazione Hardware**: Rileva e utilizza automaticamente la GPU (tramite CUDA per NVIDIA o MPS per Apple Silicon) se disponibile, altrimenti si affida alla CPU per garantire le massime performance.

## 🛠️ Installazione

Per eseguire l'applicazione e i relativi script di addestramento, si consiglia di utilizzare un ambiente virtuale per gestire le dipendenze. Di seguito sono riportate le istruzioni per l'installazione tramite **Conda**.

1.  **Clona il repository**:
    ```bash
    git clone https://github.com/danielelozzi/avian.git
    cd Avian
    ```

2.  **Installa Conda**:
    Se non hai Conda, puoi installare [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (una versione minimale di Anaconda). Scegli l'installer appropriato per il tuo sistema operativo.
    Per sfruttare l'accelerazione GPU con schede NVIDIA, assicurati di aver installato il [driver NVIDIA](https://www.nvidia.it/Download/index.aspx?lang=it) e il [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) compatibili con la versione di PyTorch specificata in `requirements.txt`.

3.  **Crea e attiva l'ambiente Conda**:
    Crea un nuovo ambiente virtuale (es. chiamato `avian_env`) con una versione di Python compatibile (es. 3.9) e attivalo.
    ```bash
    conda create --name avian_env python=3.9 -y
    conda activate avian_env
    conda install pip
    ```

4.  **Installa le dipendenze**:
    Con l'ambiente attivo, installa i pacchetti Python necessari tramite `pip`.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Scarica il modello pre-addestrato**:
    Il modello di default (`yolov8nseg_avian_100epoche.pt`) è già referenziato nell'applicazione. Se hai addestrato un tuo modello personalizzato, puoi selezionarlo tramite l'interfaccia grafica.

## 🖥️ Utilizzo dell'Applicazione

1.  **Avvia l'applicazione**:
    ```bash
    python main.py
    ```
2.  **Seleziona l'Immagine di Input**: Usa il pulsante "Sfoglia..." nella sezione 1.
3.  **(Opzionale) Seleziona un Modello**: Se vuoi usare un modello diverso da quello di default, selezionalo nella sezione 2.
4.  **(Opzionale) Configura il Preprocessing**: Nella sezione 3, scegli le opzioni di preprocessing desiderate.
5.  **Configura l'Analisi**: Assicurati che "Esegui Conteggio Cellule" sia spuntato (sezione 4).
6.  **Specifica il Percorso di Output**: Un percorso di default viene generato automaticamente, ma puoi cambiarlo nella sezione 5.
7.  **(Opzionale) Personalizza la Visualizzazione**: Regola spessore e dimensione font nella sezione 6.
8.  **Avvia l'Analisi**: Clicca su "Avvia Analisi" e attendi il completamento. I log e un riepilogo dei conteggi appariranno nella finestra principale.

## 🧠 Addestramento del Modello (Opzionale)

La cartella `dataset_marburgo` contiene gli script per creare un dataset in formato YOLOv8 e addestrare un modello di segmentazione personalizzato.

1.  **Prepara la struttura delle cartelle**:
    Prima di eseguire qualsiasi script, organizza il tuo dataset (scaricabile da qui: [dataset marburgo](https://data.uni-marburg.de/entities/dataset/c78489de-e08c-4818-800b-1f182aa2e631)). All'interno della cartella `dataset marburgo`, crea una sottocartella chiamata `dataset_segmentation`. Questa cartella deve contenere le immagini e le etichette (in formato YOLO) suddivise in `train`, `val` e (opzionalmente) `test`.

    La struttura finale dovrà essere simile a questa:
    ```
    Avian/
    └── dataset_marburgo/
        ├── dataset_segmentation/
        │   ├── train.json
        │   ├── val.json
        │   ├── train/      # Cartella con le immagini di training
        │   │   ├── image1.jpg
        │   │   └── ...
        │   └── val/        # Cartella con le immagini di validazione
        │       ├── image2.jpg
        │       └── ...
        ├── prepare_dataset.py
        ├── create_yaml.py
        └── train_yolo.py
    ```

2.  **Esegui gli script di preparazione e addestramento**:
    Assicurati che il tuo ambiente Conda (`avian_env`) sia attivo. Posizionati nella cartella `dataset marburgo` ed esegui gli script in questa sequenza:
    ```bash
    cd dataset_marburgo
    # 1. Converte il dataset da formato COCO a YOLO, splittandolo in train/val/test
    python prepare_dataset.py
    # 2. Crea il file .yaml di configurazione per YOLO
    python create_yaml.py
    # 3. Avvia il fine-tuning del modello con data augmentation
    python train_yolo.py
    ```

3.  **Trova e utilizza il modello addestrato**:
    Al termine dell'addestramento, lo script `train_yolo.py` salverà due versioni del modello:
    -   `runs/segment/yolov8n_avian_blood_seg_augmented/weights/best.pt`: Il modello con le migliori performance sul set di validazione, salvato nella cartella dei run di Ultralytics.
    -   `yolov8nseg_avian.pt`: Una copia del modello migliore, salvata nella cartella `dataset_marburgo` per un accesso più comodo. Spostala nella directory principale per usarla con l'app.

4.  **Valuta le performance del modello (Consigliato)**:
    Per ottenere metriche quantitative (Precision, Recall, mAP) sul set di test e generare una matrice di confusione, esegui lo script di valutazione. Assicurati che il modello addestrato (`yolov8nseg_avian.pt`) sia stato spostato nella cartella principale del progetto (`Avian/`).
    ```bash
    # Esegui dalla cartella "dataset_marburgo"
    python evaluate_model.py
    ```
    Questo script caricherà il modello e calcolerà le sue performance sul set di test, fornendo una valutazione oggettiva della sua efficacia e salvando la matrice di confusione come immagine.

### 📊 Spiegazione delle Metriche di Valutazione

Lo script `evaluate_model.py` (e il processo di validazione durante l'addestramento) utilizza metriche standard per valutare le performance del modello di segmentazione. Ecco una breve spiegazione delle più importanti:

-   **Intersection over Union (IoU)**: È la metrica fondamentale per la segmentazione. Misura la sovrapposizione tra la maschera predetta dal modello e la maschera reale (ground truth). Viene calcolata come `(Area di Intersezione) / (Area di Unione)`. Un valore di 1.0 indica una sovrapposizione perfetta.

-   **Precision (P)**: Indica la precisione delle predizioni. Risponde alla domanda: "Di tutte le cellule che il modello ha identificato, quante erano corrette?". Un valore alto significa che il modello ha pochi falsi positivi.

-   **Recall (R)**: Indica la capacità del modello di trovare tutte le cellule rilevanti. Risponde alla domanda: "Di tutte le cellule realmente presenti nell'immagine, quante ne ha trovate il modello?". Un valore alto significa che il modello ha pochi falsi negativi (non ha "mancato" molte cellule).

-   **mAP50 (mean Average Precision @ 50%)**: È una delle metriche principali per giudicare un modello. Rappresenta la media della precisione calcolata con una soglia di IoU del 50%. In pratica, se la sovrapposizione (IoU) tra la predizione e la maschera reale è superiore al 50%, la predizione è considerata un successo. Un valore più vicino a 1.0 è migliore.

-   **mAP50-95 (mean Average Precision @ 50-95%)**: È una metrica più severa e completa. È la media delle performance (mAP) calcolata su diverse soglie di IoU, dal 50% al 95%. Un punteggio alto qui indica che il modello non solo trova le cellule, ma le segmenta anche con un contorno molto preciso.

Queste metriche vengono calcolate sia per i riquadri di delimitazione (`Box`) che per le maschere di segmentazione (`Mask`). Per questo progetto, le metriche relative alla **maschera (`Mask`)** sono le più importanti, in quanto misurano la qualità della segmentazione.