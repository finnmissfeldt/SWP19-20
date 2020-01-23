# SWP19-20
FH Wedel Softwareproject WS19/20

notwendige Versionen:
    dlib-version: 19.19.0
    cv2-version: 4.1.2




*********************************************
**** Anmerkungen zur Verzeichnisstruktur: ***
*********************************************

main:
    Ersetzt Gesichter aus einem übergebenen Video mit Gesichtern aus einem angegebenen Ordner.

detection:
    Beinhaltet alles was sich mit der Erkennung von Gesichtern beschäftigt.
    Download notwendig: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 (entpacken und in den Ordner "detection/models"     mit dem Namen "shape_predictor_68_face_landmarks.dat")


generation:
    Beinhaltet alles was sich mit der Generierung von Gesichtern beschäftigt.


generation/generation_stylegan:
    Beinhaltet alles was sich mit dem Ansatz beschäftigt ein kompatibles Gesicht
    zu erzeugen, indem das Latentspace am Eingang des Stylegan-Generators (von NVIDIA)
    modifiziert wird.
    Kompatibel heißt hier, dass zu einem gegebenen Gesicht ein Gesicht gefunden
    wird, durch das man das Gegebene ersetzen kann.
    Beispiel: Man würde nicht das Gesicht eines schwarzen alten Mannes durch das
    eines weiblichen hellhäutigen Babys ersetzen.

    Da sich die Merkmale des zu generierenden Gesichtes nicht so einfach im Latentspace
    wiederfinden lassen, befindet ich in diesem Unterordner ein neuronales Netz,
    das zu einem Gegebenen Gesicht ein Latentspace finden soll, mit welchem das
    NVIDIA-Stylegan ein kompatibles Gesicht ezeugt.


generation/generation_stylegan/doc:
    Sammelordner für Dokumentation (-sfragmente) zum Thema "Generation via Stylegan"


generation/generation_stylegan/training_data_generator.py:
    Python-programm zum Erzeugen von Trainingsdaten zum Trainineren des Neuronalen
    Netzes, welches zu einem gegeben Gesicht den Latentspace liefert.


generation/generation_stylegan/facegeneration.py:
    Python-programm, dass funktionen bereitstellt um mit dem NVIDIA-Stylegan
    (gezielt) Gesichter zu erzeugen.
