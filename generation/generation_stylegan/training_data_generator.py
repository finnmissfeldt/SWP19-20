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

import facegeneration as fg
import chunkSerializer as cs
warnings.filterwarnings("ignore", category=DeprecationWarning)

IMAGE_RESOLUTION = 16           # Example IMAGE_RESOLUTION = 16 means resulting resolution = 16x16
AMOUNT_OF_SAMPLES = 50      # The amount of Faces that shall be generated.


# Creates a number of Faces...
def createLatentFaceMappingJson():
    pretrained_gan = fg.init()
    t_start = time.clock()

    for i in range(0, AMOUNT_OF_SAMPLES):

        # Create new random Latentspace seeded by system clock.
        latent = np.random.randn(512)

        # Generate new Face
        image_data = fg.generate(latent, pretrained_gan)
        img = PIL.Image.fromarray(image_data, 'RGB')
        if IMAGE_RESOLUTION > 0:
            img = img.resize((IMAGE_RESOLUTION, IMAGE_RESOLUTION), PIL.Image.BILINEAR)
        #image_data = img.getdata()
        cs.addData((latent, img))
        # Save image to file. (Resolution is reduced in this step)
        #fg.saveImage(image_data, str(i) + '.png', IMAGE_RESOLUTION)


        # Save actual latent-data to json-file.
        file_path = os.path.join("results/", str(i) + ".json")
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


# Create a lot of faces and store its latentspace in json.
createLatentFaceMappingJson()
