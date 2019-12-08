import numpy as np
import sys
import os

# Konstanten
DIR_PATH = "training_data/"  # The Path where Serializationdata is stored.
CHUNK_SIZE = 10000    # Amount of Datasets to write in Buffer until they are stored.

# Variablen / Speicher
buffer = []
save_actual_chunk_id = 0
load_chunk_id = 0
FILE_COUNTER = 0


if not os.path.exists(DIR_PATH):
    os.makedirs(DIR_PATH)


def addData(data):
    buffer.append(data)
    # Wenn puffer voll, dann schreibe in Datei und leere puffer.
    if (len(buffer) >= CHUNK_SIZE):
        flush()

# Leert den Puffer und schreibt den Inhalt vorher in Datei.
def flush():
    global save_actual_chunk_id
    FILE_NAME = DIR_PATH + '/' + str(save_actual_chunk_id)
    np.save(FILE_NAME, buffer)
    save_actual_chunk_id = save_actual_chunk_id + 1
    buffer.clear()

def getChunkSize():
    return CHUNK_SIZE


# NOTE Train and save trained Neuranal Network,
# Add Flag at File Name, "_" underscore before file names, to indicate that the file was already processed
#
# gets the next training data from .pkl file
def getNextArrayFromFile(index):
    global DataSet
    global FILE_COUNTER
    path = DIR_PATH + str(FILE_COUNTER) + ".npy"
    DataSet = np.load(path, allow_pickle=True)

    if (index >= CHUNK_SIZE):
        FILE_COUNTER += 1
        index = index % CHUNK_SIZE

    return np.asanyarray(DataSet[index])

"""
# Testen...
for i in range(0, 100):
    dataset = (np.random.randn(512), np.random.randn(16, 16, 3))
    addData(dataset)
"""
