import json
import PIL.Image
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D, Conv2D, InputLayer
import numpy as np
import os.path as path
import math
import os, os.path


# Constants
IMAGE_RESOLUTION = 16           # Resulting resolution = IMAGE_RESOLUTION x IMAGE_RESOLUTION
#INPUT_DIMENSION = IMAGE_RESOLUTION * IMAGE_RESOLUTION * 3   # Achtung muss zur Auflösung / Farbwerten der Trainingsbilder passen.
OUTPUT_DIMENSION = 512          # Achtung muss zur größe des Latent-Arrays aus den Trainingsjsons passen.
TRAINING_DATA_DIR = "./training_data/16x16_100k/"
AMOUNT_EPOCHS = 1
BATCH_SIZE = 10
MAX_AMOUNT_OF_TRAINING_DATA_CHUNKS = 2

model = Sequential()


def init():
    assert IMAGE_RESOLUTION == 16, "Neural net has to be adjusted for this Imageresolution."
    assert OUTPUT_DIMENSION == 512, "Neural net has to be adjusted for this output dimension."

    model.add(InputLayer(input_shape=(16, 16, 3)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
   # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(512, activation='sigmoid')) # 512 = OUTPUT_DIMENSION

    model.compile(optimizer='rmsprop', loss='mse')  # For a mean squared error regression problem

   #print("=======> OUT: ", model.layers[0].output)
    print("Model: ", model.summary())

# ! deprecated !
# only for resize
# Loads image from results folder and parses to numpy array
# @param path, the Path of the Images
def getArrayFromImage(img, autoresize=False):
    if autoresize:
        img = img.resize((IMAGE_RESOLUTION, IMAGE_RESOLUTION), PIL.Image.BILINEAR)
    return np.asarray(img)



def train():
    # Initialize Datasets
    for i in range(0, MAX_AMOUNT_OF_TRAINING_DATA_CHUNKS):
        path = TRAINING_DATA_DIR + str(i) + '.npy'
        if not os.path.exists(path):
            break;
        data = np.load(path, allow_pickle=True)
        model.fit(data[1], data[0], epochs=AMOUNT_EPOCHS, batch_size=BATCH_SIZE)


# Map Standard Normal distribution to a Space between 0 and 1 (using sigmoid (base 2))
def latent_to_signal(input):
    for i in range(0, len(input)):
        input[i] = 1.0 / (1.0 + 2**(-input[i]))

# Map distribution between 0 and 1 to Normal distribution (inverse sigmoid (base 2))
def signal_to_latent(input):
    for i in range(0, len(input)):
        assert input[i] >= 0 and input[i] <= 1, "Error. Input out of range. Input: " + str(input[i])
        input[i] = input[i] + 0.00001
        input[i] = - math.log(1.00002 / input[i] - 1, 2)





def uint8_to_float_image(input):
    w = len(input)
    assert w == len(input[0]), "Image has to be quadratic"
    out = np.ndarray(shape=(w, w, 3), dtype=np.float32)

    for i in range(0, w):
        for j in range(0, w):
            out[i][j][0] = 1.0 * input[i][j][0] / 255.0
            out[i][j][1] = 1.0 * input[i][j][1] / 255.0
            out[i][j][2] = 1.0 * input[i][j][2] / 255.0
    return out


# Same as generate(img_data), but load image-data from file.
# @param path The path to the Image.
# @return The latentvector
def generate(path):
    return genrate(getArrayFromImage(path, True))


# Generate (Predict) Latentspace for given Image
# The Resolution of the Image is irrelevant as it will be resized to the required
# resolution automatically. (The file will not be changed.)
# @param img_data 3-D-Array containing img data. Shape = (RESO_X, RESO_Y, COLORS)
# @return The latentvector
def generate(img_data):
    assert len(img_data) == IMAGE_RESOLUTION
    assert len(img_data[0]) == IMAGE_RESOLUTION
    assert len(img_data[0][0]) == 3
    input = np.ndarray(shape=(1, IMAGE_RESOLUTION, IMAGE_RESOLUTION, 3))
    input[0] = img_data
    output_generated = model.predict(input, batch_size=1, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
#    print("Signals: ", output_generated[0][0], output_generated[0][1], output_generated[0][2],\
#                             output_generated[0][3], output_generated[0][4], output_generated[0][5])
    signal_to_latent(output_generated[0])
    return output_generated[0]

init()
train()
