""" Diese Datei ist dazu da mein (Eriks) neuronales Netz aus latent_finder.py zu
    testen. Das neuronale Netz soll zu einem gegebenen Bild ein Latentspace finden,
    das dem NVIDIA-Stylegan als Eingabe dient. Dieses Latentspace soll so gewählt
    sein, dass das aus dem Stylegan resultierende Bild dem Ausgansbild ähnlich ist.
    Diese Ähnlichkeit soll bei diesem Ansatz vom Menschen überprüft werden.
    (Das Training des Netzes passiert vollautomatisch.)
    Um das überprüfen zu ermöglichen, werden im Ordner results/evaluation/ eine
    Reihe Bilder erzeugt. Die Eingangsbilder nennen sich X_in.png, die Ergebnisbilder
    X_out.png (X ist durch eine Fortlaufende Zahl zu ersetzten).
    Nun kann man als Mensch überprüfen ob diese Bilder "kompatibel" sind.
    "kompatibel" heißt hier, dass das Endprogramm in einem Foto/Video das Eingangsbild (X_in.png)
    durch das Ausgangsbild ersetzen darf, ohne dass es komisch aussieht.
    Beispiele:
        - Das Gesicht einer weißen Frau dürfte nicht ersetzt werden, durch das eines
          schwarzen Mannes.
        - Das Gesicht eines sehr alten Mannes dürfte nicht ersetzt werden, durch
          das eine Babys"""

import numpy as np
# Das folgenden Modul muss sich selber beim importieren initialisieren. (init() aufrufen)
import latent_finder_neural_network as lf
import facegeneration as fg
import PIL.Image



# Achtung es müssen mindestens genauso viele Trainingsdatensätze vorliegen
# wie hier Vergleichsdaten erzeugt werden.
# Hier werden (auch wenn es etwas geschummelt ist) die Trainingsdaten
# wiederbenutzt. Beim "Endtest", sollten nochmal gesondert Daten erzeugt werden.


# Konstanten
AMOUNT_OF_EVAL_SETS = 20
RESULT_DIR = "result/"  # Muss auf / enden.


# Variablen / Speicher
input_latents = []
output_latents = []
fg_gan = fg.init()



avg_loss = 0
for i in range(0, AMOUNT_OF_EVAL_SETS):

    # Step 1: Create random latentspace
    input_latents.append(np.random.randn(512))

    # Step 2: Generate Face-image-data for given Latent, using Nvidia-Stylegan.
    img_data = fg.generate(input_latents[i], fg_gan)
    img = PIL.Image.fromarray(img_data, 'RGB') # Redundante datenhaltung für performance
    img_data = np.array(img.resize((lf.IMAGE_RESOLUTION, lf.IMAGE_RESOLUTION), PIL.Image.BILINEAR)) # img_data von 1024x1024 auf passende Auflsung...

    # Step 3: Use our latent_finder_neural_network to find latentspace for that img.
    output_latents.append(lf.generate(img_data))

    # Step 4: Generate and save image, from just generated latentspace.
    fg.saveImage(fg.generate(fg_gan, output_latents[i]), RESULT_DIR + str(i) + '_out.png')


    # Calc and print loss
    loss = 0
    for j in range(0, 512):
        delta = new_latents[i][j] - original_latents[i][j]
        loss = loss   +   delta * delta / 512.0
    print("MSE Loss in Latentspace: ", loss)
    avg_loss = avg_loss + loss / AMOUNT_OF_EVAL_SETS

print("====== Average loss:   ", avg_loss, "   ======")
