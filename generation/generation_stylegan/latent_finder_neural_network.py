import json
import PIL.Image
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D, Conv2D, InputLayer
import numpy as np
import os.path as path
import math

# Constants
IMAGE_RESOLUTION = 16           # Resulting resolution = IMAGE_RESOLUTION x IMAGE_RESOLUTION
#INPUT_DIMENSION = IMAGE_RESOLUTION * IMAGE_RESOLUTION * 3   # Achtung muss zur Auflösung / Farbwerten der Trainingsbilder passen.
OUTPUT_DIMENSION = 512          # Achtung muss zur größe des Latent-Arrays aus den Trainingsjsons passen.
TRAINING_DATA_DIR = "./results/reso16/"
AMOUNT_EPOCHS = 1
BATCH_SIZE = 50
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

    print("Model: ", model.summary())

def getArrayFromImage(path, autoresize=False):
    img = PIL.Image.open(path)
    if autoresize:
        img = img.resize((IMAGE_RESOLUTION, IMAGE_RESOLUTION), PIL.Image.BILINEAR)
    return np.asarray(img)


def train():
    # Test how many Faces and Laten-jsons are existing.
    amount_of_samples = 0
    while True:
        if (path.exists(TRAINING_DATA_DIR + str(amount_of_samples) + '.json') and path.exists(TRAINING_DATA_DIR + str(amount_of_samples) + '.png')):
            amount_of_samples = amount_of_samples + 1
        else:
            break
    print("Amount of Datasets: ", amount_of_samples)
    if (amount_of_samples <= 0):
        print("Caution no Data red in! This will result in an Error later on! Exiting now.")
        exit()

    # Initialize Datasets
    input = np.ndarray(shape=(amount_of_samples, IMAGE_RESOLUTION, IMAGE_RESOLUTION, 3))
    output_expected = np.ndarray(shape=(amount_of_samples, OUTPUT_DIMENSION))

    # Fill Datasets from Files.
    for i in range(0, amount_of_samples):

        # Load inputdata (from image)
        input[i] = uint8_to_float_image(getArrayFromImage(TRAINING_DATA_DIR + str(i) + '.png'))

        # Load outputdata (from json)
        file = open(TRAINING_DATA_DIR + str(i) + '.json', "r")
        file_data = file.read()
        file.close()
        output_expected[i] = json.loads(file_data)
        latent_to_signal(output_expected[i])

    # Train
    model.fit(input, output_expected, epochs=AMOUNT_EPOCHS, batch_size=BATCH_SIZE)
    print("=======> OUT: ", model.layers[0].output)


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


# Generate (Predict) Latentspace for given Image
# The Resolution of the Image is irrelevant as it will be resized to the required
# resolution automatically. (The file will not be changed.)
# @param path The path to the Image.
# @return The latentvector
def generate(path):
    input = np.ndarray(shape=(1, IMAGE_RESOLUTION, IMAGE_RESOLUTION, 3))
    input[0] = getArrayFromImage(path, True)
    print
    output_generated = model.predict(input, batch_size=1, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
    print("Signals: ", output_generated[0][0], output_generated[0][1], output_generated[0][2],\
                             output_generated[0][3], output_generated[0][4], output_generated[0][5])
    signal_to_latent(output_generated[0])

    return output_generated[0]

init()
train()
