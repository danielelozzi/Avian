# AvianV2

# Preprocessing

Pipeline di preprocessing per immagini di vetrini da microscopio. L'obiettivo è isolare, pulire e ritagliare l'area di interesse (il campione biologico) dall'immagine grezza, preparandola per successive analisi come il conteggio di cellule o l'estrazione di feature.

## Funzionalità

Il cuore del progetto è il file `cell_preprocessing.py`, che contiene la logica per il trattamento delle immagini. La funzione principale, `preprocess_image`, esegue una serie di passaggi per pulire l'immagine.

### Pipeline di Preprocessing

La pipeline implementata nella funzione `preprocess_image` segue questi passaggi:

1.  **Rimozione Canale Alpha**: Se l'immagine di input è in formato RGBA (4 canali), il canale alpha viene rimosso per lavorare solo con i canali RGB.

2.  **Miglioramento del Contrasto**:
    -   Viene analizzato il canale rosso dell'immagine per determinare le soglie di intensità ottimali tramite l'algoritmo `threshold_multiotsu`.
    -   L'intensità dell'intera immagine viene riscalata (`rescale_intensity`) per migliorare il contrasto, facendo risaltare il campione rispetto allo sfondo.

3.  **Isolamento dell'Area del Campione**:
    -   Viene creata un'immagine binaria basata sul canale rosso (precedentemente migliorato) per separare lo sfondo (il vetrino) dal campione.
    -   Tramite `skimage.measure.regionprops_table`, vengono identificate e analizzate le diverse aree. Viene selezionata la regione più grande e meno eccentrica, che corrisponde all'area circolare del campione sul vetrino.

4.  **Mascheratura Circolare**:
    -   Viene generata una maschera circolare basata sul centro e sul raggio dell'area del campione identificata al punto precedente.
    -   Questa maschera viene applicata all'immagine per eliminare qualsiasi rumore o artefatto presente al di fuori dell'area di interesse.

5.  **Ritaglio (Cropping)**:
    -   Infine, l'immagine viene ritagliata utilizzando le coordinate del rettangolo di delimitazione (bounding box) dell'area del campione. Questo rimuove lo sfondo nero in eccesso e restituisce un'immagine focalizzata solo sul campione preprocessato.

## Dipendenze

Per utilizzare questo modulo, assicurati di avere installato le seguenti librerie Python:

-   `numpy`
-   `pandas`
-   `scikit-image`

Puoi installarle tramite pip:
```bash
pip install numpy pandas scikit-image
```

## Utilizzo

Per utilizzare la funzione di preprocessing, importa la funzione `preprocess_image` dal file `cell_preprocessing.py` e passale l'immagine come array NumPy.

```python
import numpy as np
from PIL import Image
from cell_preprocessing import preprocess_image
import matplotlib.pyplot as plt

# Carica l'immagine
image_path = 'path/to/your/slide_image.jpg'
input_image = np.array(Image.open(image_path))

# Applica il preprocessing
processed_tensor = preprocess_image(input_image)

# Ora `processed_tensor` contiene l'immagine ritagliata e pulita
print(f"Dimensioni originali: {input_image.shape}")
print(f"Dimensioni dopo il preprocessing: {processed_tensor.shape}")

# Visualizza il risultato
plt.imshow(processed_tensor)
plt.title("Immagine Preprocessata")
plt.axis('off')
plt.show()
```
