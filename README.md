# Avian - Analizzatore di Immagini di Vetrini

**Avian** è un'applicazione desktop sviluppata in Python per l'analisi di immagini di campioni biologici su vetrini, con un focus specifico sul conteggio e la classificazione di cellule ematiche aviarie.

L'applicazione utilizza un modello di deep learning in formato ONNX per la rilevazione degli oggetti (cellule) e offre un'interfaccia grafica semplice per automatizzare il processo di analisi.

## Funzionalità

- **Interfaccia Grafica Semplice**: Un'interfaccia utente intuitiva costruita con Tkinter per guidare l'utente attraverso il processo.
- **Selezione Dinamica dei File**: Permette di selezionare facilmente l'immagine di input, il percorso di output e il modello di IA da utilizzare.
- **Fasi di Analisi Configurabili**:
  - **Preprocessing**: Isola automaticamente l'area di interesse circolare del campione sul vetrino, migliorando il contrasto e ritagliando l'immagine.
  - **Conteggio Cellule**: Utilizza un modello ONNX per rilevare, classificare e contare le cellule (es. eritrociti, trombociti, linfociti).
- **Output Multipli**:
  - **Immagine Annotata**: Salva una nuova immagine con i riquadri (bounding box) disegnati attorno alle cellule rilevate, etichettati con classe e confidenza.
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
    pip install numpy pandas scikit-image Pillow onnxruntime matplotlib
    # Per supporto GPU NVIDIA:
    # pip install onnxruntime-gpu
    ```

3.  **Scarica il modello**:
    Assicurati che il file del modello ONNX (es. `efficientNet_B0.onnx`) sia presente nella stessa cartella dello script `main.py`, oppure selezionalo tramite l'interfaccia.

## Utilizzo

1.  **Avvia l'applicazione**:
    ```bash
    python main.py
    ```
2.  **Seleziona l'Immagine di Input** tramite il pulsante "Sfoglia...".
3.  **(Opzionale) Seleziona un Modello di IA** se vuoi usarne uno diverso da quello di default.
4.  **Scegli le Fasi di Analisi** (Preprocessing e/o Conteggio).
5.  **Specifica il Percorso di Output** per l'immagine analizzata.
6.  **Clicca su "Avvia Analisi"** e attendi il completamento. I risultati e i log appariranno nella finestra principale.