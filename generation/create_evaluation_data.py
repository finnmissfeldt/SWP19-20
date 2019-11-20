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

import json
import random
# Das folgenden Modul muss sich selber beim importieren initialisieren. (init() aufrufen)
import latent_finder_v3 as lf




# Achtung es müssen mindestens genauso viele Trainingsdatensätze vorliegen
# wie hier Vergleichsdaten erzeugt werden.
# Hier werden (auch wenn es etwas geschummelt ist) die Trainingsdaten
# wiederbenutzt. Beim "Endtest", sollten nochmal gesondert Daten erzeugt werden.

AMOUNT_OF_EVAL_SETS = 100
original_latents = []
new_latents = []


for i in range(0, AMOUNT_OF_EVAL_SETS):
    file = open(lf.TRAINING_DATA_DIR + str(i) + '.json', "r")
    original_latents.append(json.loads(file.read()))
    file.close()
    new_latents.append(lf.generate(lf.TRAINING_DATA_DIR + str(i) + '.png').copy())
    #new_latents.append(original_latents[i].copy())
    #for j in range(0, 200):
    #    new_latents[i][j] = random.random()

# Das folgenden Modul muss sich selber beim importieren initialisieren. (init() aufrufen)
# Das folgende Modul kann erst hier importiert werden, da sonst das andere nicht mehr nutzbar ist.
import eriks_training_data_generator as trainer

avg_loss = 0
for i in range(0, AMOUNT_OF_EVAL_SETS):
    trainer.saveImage(trainer.generate(trainer.trained_ki, original_latents[i]), 'evaluation/' + str(i) + '_in.png')
    trainer.saveImage(trainer.generate(trainer.trained_ki, new_latents[i]), 'evaluation/' + str(i) + '_out.png')

    # Calc and print loss
    loss = 0
    for j in range(0, 512):
        delta = new_latents[i][j] - original_latents[i][j]
        loss = loss   +   delta * delta / 512.0
    print("MSE Loss in Latentspace: ", loss)
    avg_loss = avg_loss + loss / AMOUNT_OF_EVAL_SETS

print("====== Average loss:   ", avg_loss, "   ======")
