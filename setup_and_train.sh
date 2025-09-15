#!/bin/bash

# Script per configurare l'ambiente e avviare il training del modello Avian.
# Lo script si fermerà immediatamente se un comando fallisce.
set -e

# --- Variabili di Configurazione ---
DATASET_URL="https://data.uni-marburg.de/bitstreams/b95eee88-d7a6-4700-97ec-7a0d9dea3411/download"
REPO_URL="https://github.com/danielelozzi/Avian.git"
CONDA_ENV_NAME="avian"
PROJECT_DIR="Avian"
DATASET_DIR_NAME="dataset_segmentation"

# --- 1. Download del Dataset ---
echo "--- Fase 1: Download del dataset... ---"
wget -O dataset.tar.gz "$DATASET_URL"
echo "Download completato."
echo

# --- 2. Estrazione del Dataset ---
echo "--- Fase 2: Estrazione del file TAR... ---"
# Il file è un .tar.gz, quindi usiamo l'opzione 'z' per decomprimere
tar -xzvf dataset.tar.gz
echo "Estrazione completata. La cartella '$DATASET_DIR_NAME' è stata creata."
echo

# --- 3. Download della Repository ---
echo "--- Fase 3: Clonazione della repository da GitHub... ---"
git clone "$REPO_URL"
echo "Repository clonata nella cartella '$PROJECT_DIR'."
echo

# --- 4. Posizionamento del Dataset ---
echo "--- Fase 4: Spostamento del dataset nella cartella corretta... ---"
mv "$DATASET_DIR_NAME" "$PROJECT_DIR/dataset_marburgo/"
echo "Dataset spostato in '$PROJECT_DIR/dataset_marburgo/$DATASET_DIR_NAME'."
echo

# --- 5. Installazione di Anaconda ---
echo "--- Fase 5: Installazione di Anaconda (Miniconda)... ---"
# Controlla se conda è già installato per evitare una nuova installazione
if command -v conda &> /dev/null
then
    echo "Conda è già installato. Salto l'installazione."
    CONDA_BASE_PATH=$(conda info --base)
else
    echo "Conda non trovato. Procedo con l'installazione di Miniconda."
    # Scarica l'installer di Miniconda per Linux 64-bit
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3.sh
    # Esegui l'installer in modalità batch (non interattiva)
    bash Miniconda3.sh -b -p $HOME/miniconda3
    # Rimuovi il file di installazione
    rm Miniconda3.sh
    CONDA_BASE_PATH="$HOME/miniconda3"
    echo "Installazione di Miniconda completata in '$CONDA_BASE_PATH'."
fi
echo

# Inizializza Conda per questo script
eval "$($CONDA_BASE_PATH/bin/conda shell.bash hook)"

# --- 6. Creazione dell'Ambiente Conda ---
echo "--- Fase 6: Creazione e configurazione dell'ambiente Conda '$CONDA_ENV_NAME'... ---"
# Controlla se l'ambiente esiste già
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "Ambiente '$CONDA_ENV_NAME' già esistente. Lo attivo."
else
    echo "Creazione del nuovo ambiente '$CONDA_ENV_NAME'..."
    conda create --name "$CONDA_ENV_NAME" python=3.9 -y
fi

# Attiva l'ambiente
conda activate "$CONDA_ENV_NAME"
echo "Ambiente '$CONDA_ENV_NAME' attivato."

# Installa le dipendenze
echo "Installazione delle dipendenze da requirements.txt..."
pip install -r "$PROJECT_DIR/requirements.txt"
echo "Dipendenze installate."
echo

# --- 7. Esecuzione degli Script di Training ---
echo "--- Fase 7: Avvio degli script di preparazione e addestramento... ---"
# Entra nella cartella corretta
cd "$PROJECT_DIR/dataset_marburgo"

echo "Esecuzione di 'prepare_dataset.py'..."
python prepare_dataset.py

echo "Esecuzione di 'create_yaml.py'..."
python create_yaml.py

echo "Esecuzione di 'train_yolo.py' (questa fase potrebbe richiedere molto tempo)..."
python train_yolo.py

echo
echo "--- TUTTO COMPLETATO! ---"
echo "Il processo di addestramento è terminato. Troverai il modello addestrato nella cartella '$PROJECT_DIR'."

# chmod +x setup_and_train.sh
#./setup_and_train.sh
