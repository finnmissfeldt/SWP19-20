import numpy as np


# Konstanten
DIR_PATH = "t_data"  # The Path where Serializationdata is stored.
CHUNK_SIZE = 10000    # Amount of Datasets to write in Buffer until they are stored.

# Variablen / Speicher
buffer = []
save_actual_chunk_id = 0
load_chunk_id = 0


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



# Testen...
for i in range(0, 100000):
    dataset = (np.random.randn(512), np.random.randn(16, 16, 3))
    addData(dataset)
