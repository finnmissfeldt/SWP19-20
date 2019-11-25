import os
import pickle
import numpy as np
import PIL.Image

import time
import warnings
import collections
import json
import codecs
import sys

# Nvidia Stylegan stuff
sys.path.insert(1, "nvidia_lib/")
import nvidia_lib.dnnlib as dnnlib
import nvidia_lib.dnnlib.tflib as tflib
import nvidia_lib.config as config


warnings.filterwarnings("ignore", category=DeprecationWarning)

IMAGE_RESOLUTION = 16 # Example IMAGE_RESOLUTION = 16 means resulting resolution = 16x16
AMOUNT_OF_SAMPLES = 200000           # The amount of Faces that shall be generated.


def init():
    tflib.init_tf()     # Initialize TensorFlow.
    _G, _D, Gs = pickle.load(open("nvidia_lib/original_pretrained_stylegan.pkl", "rb"))    # Load pre-trained network.
    return Gs


# @param Gs Das trainierte Netzwerk, mit welchem nun ein Bild generiert werden soll.
# @param latentSpace Das latentSpace, das zur Erzeugung genutzt werden soll.
#         (Wenn none, dann wird ein Randomvektor genutzt.)
def generate(Gs, latentSpace):
    print("Some Latents: ", latentSpace[0], latentSpace[1], latentSpace[2],\
                            latentSpace[3], latentSpace[4], latentSpace[5])
    t_start_generation = time.clock()
    latents = np.ndarray(shape=(1,512), dtype=np.float64)
    latents[0] = latentSpace;
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)  # Generate image.
    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    print("Time needed for generation: ", time.clock() - t_start_generation)
    return images[0]


# Saves the Image to file
# @param image_data The ImageData as numpy.ndarray shape=(x_resolution, y_resolution, amount_of_colors)
# @param names  The filename. (The Path relative to config.result_dir)
def saveImage(image_data, name, autoresize=False):
    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, name)
    img = PIL.Image.fromarray(image_data, 'RGB')
    if autoresize:
        img = img.resize((IMAGE_RESOLUTION, IMAGE_RESOLUTION), PIL.Image.BILINEAR)
    img.save(png_filename)


# Creates a number of Faces...
def createLatentFaceMappingJson(trained_ki):
    t_start = time.clock()

    for i in range(0, AMOUNT_OF_SAMPLES):

        # Create new random Latentspace seeded by system clock.
        latent = np.random.randn(512)

        # Generate new Face
        image_data = generate(trained_ki, latent)

        # Save image to file. (Resolution is reduced in this step)
        saveImage(image_data, str(i) + '.png', True)

        # Save actual latent-data to json-file.
        file_path = os.path.join(config.result_dir, str(i) + ".json")
        file = open(file_path, "w+")    # Open file (+ means: create if not existing)
        file.write(json.dumps(latent.tolist()))
        file.close()

        t_delta = time.clock() - t_start
        t_avg_per_sample = t_delta / (1 + i)
        print("Time stats:: Progress: ", 100 * i // AMOUNT_OF_SAMPLES,\
                "%  Time-Overall: ", t_delta,\
                "sec    Avg/sample: ", t_avg_per_sample,\
                "sec    Remaining: ", t_avg_per_sample * (AMOUNT_OF_SAMPLES - i))

    print("Full Time for generation of all mappings: ", time.clock() - t_start)



trained_ki = init()

# Create a lot of faces and store its latentspace in json.
createLatentFaceMappingJson(trained_ki)
