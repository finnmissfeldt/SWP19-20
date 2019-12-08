"""
In dieser Datei wird die Funktionalität bereitgestellt um mit dem NVIDIA-Stylegan
Gesichter zu erzeugen.
Vorraussetzungen:
    Es muss in diesem Verzeichnis das Verzeichnis nvidia_lib geben, in welchem sich
    das Stylegan befindet.

Diese Datei ist als utility gedacht und nicht dazu selbst ausgeführt zu werden.
Wenn doch einfach ein Beispielbild generiert werden soll, dann muss dises Modul
1. ... importiert werden ...
2. ... init() aufgerufen werden ...
3. ... saveImage(generate(np.random.randn(512)), "example.png") ...
Das Bild example.png liegt jetzt in dem Verzeichnis results
    """

import os
import pickle
import numpy as np
import PIL.Image
import time
import warnings
import sys

# Nvidia Stylegan stuff
sys.path.insert(1, "nvidia_lib/")
import nvidia_lib.dnnlib as dnnlib
import nvidia_lib.dnnlib.tflib as tflib


warnings.filterwarnings("ignore", category=DeprecationWarning)


def init():
    tflib.init_tf()     # Initialize TensorFlow.
    _G, _D, Gs = pickle.load(open("nvidia_lib/original_pretrained_stylegan.pkl", "rb"))    # Load pre-trained network.
    return Gs


# Erzeugt ein neues Gesicht mit Hilfe des Stylegans von NVIDIA
# @param    latentSpace Das latentSpace, das zur Erzeugung genutzt werden soll.
#           (Wenn none, dann wird ein Randomvektor genutzt.)
# @return   Die Bilddaten. Als Dreidimensionales Arrays.
#           Aulösung: 1024x1014
#           Farben: 3(rgb)
#           Farbrepräsentation: numpy.uint8 (0-255)
def generate(latentSpace, pretrained_gan):
    assert pretrained_gan != None, "Neural network is not initialized. Please call init()"
    t_start_generation = time.clock()
    latents = np.ndarray(shape=(1,512), dtype=np.float64)
    latents[0] = latentSpace;
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)  # Generate image.
    images = pretrained_gan.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    #print("Time needed for generation: ", time.clock() - t_start_generation)
    return images[0]


# (Skaliert die Auflösung und) speichert das Bild als Datei (png).
# @param image_data            The ImageData as numpy.ndarray shape=(x_resolution,
#                              y_resolution, amount_of_colors)
# @param names                 Der Dateipfad relativ zu results. (The Path
#                              relative to config.result_dir)
# @param scale_to_resolution   Die Zielauflösung nach dem "Resize". Eine "Resize
#                              wird nur vorgenommen, wenn scale_to_resolution > 0"
#                              Angenommen scale_to_resolution = 16 dann ist die
#                              resultierende Auflösung 16x16
def saveImage(image_data, name, scale_to_resolution=0):
    os.makedirs("results/", exist_ok=True)
    png_filename = os.path.join("results/", name)
    img = PIL.Image.fromarray(image_data, 'RGB')
    if scale_to_resolution > 0:
        img = img.resize((scale_to_resolution, scale_to_resolution), PIL.Image.BILINEAR)
    img.save(png_filename)
