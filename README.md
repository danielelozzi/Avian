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

Per eseguire l'applicazione, è necessario avere Python 3 installato e le seguenti librerie.

1.  **Clona il repository**:
    ```bash
    git clone <URL-del-tuo-repository>
    cd <nome-cartella-repository>
    ```

2.  **Installa le dipendenze**:
    Si consiglia di utilizzare un ambiente virtuale.
    ```bash
    python -m venv venv
    source venv/bin/activate  # Su Windows: venv\Scripts\activate
    ```
    Installa i pacchetti richiesti:
    ```bash
    pip install numpy Pillow ultralytics opencv-python
    ```

3.  **Scarica il modello**:
    Assicurati che il file del modello YOLOv8 (es. `yolov8_seg.pt`) sia presente nella stessa cartella dello script `main.py`, oppure selezionalo tramite l'interfaccia.
    Puoi addestrare un modello personalizzato usando gli script nella cartella `dataset marburgo`.

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

Nella cartella `dataset marburgo` sono presenti gli script per preparare il dataset e addestrare il modello YOLOv8 per la segmentazione.

1.  **Prepara il dataset**:
    ```bash
    cd "dataset marburgo"
    python prepare_dataset.py
    ```
2.  **Crea il file di configurazione YAML**:
    ```bash
    python create_yaml.py
    ```
3.  **Avvia l'addestramento**:
    ```bash
    python train_yolo.py
    ```
Il modello addestrato (`best.pt`) si troverà nella cartella `runs/segment/yolov8n_avian_blood_seg/weights/`.