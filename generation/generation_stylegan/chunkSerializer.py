import numpy as np
import sys
import os

# Konstanten
DIR_PATH = "training_data"  # The Path where Serializationdata is stored.
CHUNK_SIZE = 10    # Amount of Datasets to write in Buffer until they are stored.

# Variablen / Speicher
buffer = []
save_actual_chunk_id = 0
load_chunk_id = 0


if not os.path.exists(DIR_PATH):
    os.makedirs(DIR_PATH)


def addData(data):
    print("Type: ", type(data))
    print("Type: ", type(data[0]), " ", type(data[1]))
    print("AddData: Size of Data", sys.getsizeof(data))
    buffer.append(data)
    # Wenn puffer voll, dann schreibe in Datei und leere puffer.
    if (len(buffer) >= CHUNK_SIZE):
        flush()

# Leert den Puffer und schreibt den Inhalt vorher in Datei.
def flush():
    print("Flush: Size of Buffer", sys.getsizeof(buffer))
    global save_actual_chunk_id
    FILE_NAME = DIR_PATH + '/' + str(save_actual_chunk_id)
    np.save(FILE_NAME, buffer)
    save_actual_chunk_id = save_actual_chunk_id + 1
    buffer.clear()


"""
# Testen...
for i in range(0, 100):
    dataset = (np.random.randn(512), np.random.randn(16, 16, 3))
    addData(dataset)
"""
