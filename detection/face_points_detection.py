#! /usr/bin/env python
import dlib
import numpy as np

PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
## Gesichts- und Punkteerkennung
def face_points_detection(img, bbox:dlib.rectangle):
    # Gibt die Landmarks/gesicht aus der Box zurück
    shape = predictor(img, bbox)

    # Iteriert über die 69 Landmarks und konvertiert diese
    # zu einem Tupel aus (x, y)-Koordinaten
    coords = np.asarray(list([p.x, p.y] for p in shape.parts()), dtype=np.int)

    # Gibt die (x, y)-Koordianten zurücken
    return coords
