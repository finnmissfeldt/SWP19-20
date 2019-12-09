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

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Konstanten
IMAGE_RESOLUTION = 16           # Example IMAGE_RESOLUTION = 16 means resulting resolution = 16x16
AMOUNT_OF_SAMPLES = 100      # The amount of Faces that shall be generated.
CHUNK_AMOUNT = 3
DIR_PATH = "training_data/"  # The Path where Serializationdata is stored.

# Variablen / Speicher
save_actual_chunk_id = 0


if not os.path.exists(DIR_PATH):
    os.makedirs(DIR_PATH)

CountOfFiles = len([name for name in os.listdir(DIR_PATH)])

tempBufferSize = int(AMOUNT_OF_SAMPLES/CHUNK_AMOUNT)
tempLatent = np.empty(shape=(tempBufferSize, 512))
tempImg = np.ndarray(shape=(int(AMOUNT_OF_SAMPLES/CHUNK_AMOUNT), IMAGE_RESOLUTION, IMAGE_RESOLUTION, 3))


# Creates a number of Faces...
def createDataMapping():

    pretrained_gan = fg.init()
    t_start = time.clock()

    for i in range(0, AMOUNT_OF_SAMPLES):
        index = i % tempBufferSize
        # Create new random Latentspace seeded by system clock.
        latent = np.random.randn(512)
        tempLatent[index] = latent
        # Generate new Face
        image_data = fg.generate(latent, pretrained_gan)
        img = PIL.Image.fromarray(image_data, 'RGB')
        if IMAGE_RESOLUTION > 0:
            img = img.resize((IMAGE_RESOLUTION, IMAGE_RESOLUTION), PIL.Image.BILINEAR)
            tempImg[index] = np.array(img)

        if index == (tempBufferSize)-1:
            saveData(tempLatent, tempImg)

        t_delta = time.clock() - t_start
        t_avg_per_sample = t_delta / (1 + i)
        print("Time stats:: Progress: ", 100 * i // AMOUNT_OF_SAMPLES,\
                "%  Time-Overall: ", t_delta,\
                "sec    Avg/sample: ", t_avg_per_sample,\
                "sec    Remaining: ", t_avg_per_sample * (AMOUNT_OF_SAMPLES - i))

    print("Full Time for generation of all mappings: ", time.clock() - t_start)




def saveData(latenData, imgData):
    global save_actual_chunk_id
    tempArr = np.ndarray(2, dtype=np.ndarray)
    tempArr[0] = latenData
    tempArr[1] = imgData
    FILE_NAME = DIR_PATH + '/' + str(save_actual_chunk_id)
    np.save(FILE_NAME, tempArr)
    save_actual_chunk_id += 1



# Create a lot of faces and store its latentspace in json.
createDataMapping()
