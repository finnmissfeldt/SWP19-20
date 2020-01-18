import cv2
from dnn_detector import dnn_detector
from dlib_detector import detector_dlib
import imutils
import argparse

"""
Verarbeitet das uebergene Video Frame fuer Frame indem Gesichter erkannt 
und durch Rechtecke angezeigt werden.
Zu Gesichtern die mit dem Dlib-frontal-face-detector erkannt werden, werden
68-markente Punkte des Gesichtes angezeigt.
Separat werden durch ein vortrainiertes KI-Model von OpenCV weitere Gesichter erkannt,
da Dlib leider nicht alle erkennt.
"""

# Parameter der Kommandozeile parsen
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="Pfad für das Input-Video")
ap.add_argument("-o", "--output", type=str,
	help="Output-Video")
args = vars(ap.parse_args())

    

#%%
def run_video(video):
    # Video-Stream initialisieren
    if(args["output"] is None):
        print("[INFO] Kein Output angegeben. Video wird nicht gespeicher")
        
    print("[INFO] Video-Stream wird gestartet...")
    print("[INFO] Taste 'q' drücken um zu beenden")
    
    vs = cv2.VideoCapture(video)
    writer = None
    
    # Iteration ueber die Frames des Videos
    while True:
    	# naechstes Frame auslesen
        (grabbed, frame) = vs.read()
    
    	# Ende des Videos abfangen
        if frame is None:
            break
    
    	# Framegroesse veraendern 
        frame = imutils.resize(frame, width=1000)

    	# videowriter initialisieren
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                (frame.shape[1], frame.shape[0]), True)
    
        
        # Gesichter erkennen mit dem DNN-Modul (gruene Rechtecke)
        dnn_detector.detect_faces(frame)
        # Gesichter erkennen (rote Rechtecke) und fuer jedes Gesicht landmarks zeichnen mit dlib (rote Punkte)
        dlib_detector.detect_landmarks(frame)

        # Output schreiben
        if writer is not None:
            writer.write(frame)
    
    	# Outputframe anzeigen
        #cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", frame)
        
        # auf Tastatureingabe reagieren
        key = cv2.waitKey(1) & 0xFF

    	# beenden wenn Taste q gedrueckt wurde
        if key == ord("q"):
            break
    

    
    # writer schliessen
    if writer is not None:
    	writer.release()
    
    # aufraeumen
    cv2.destroyAllWindows()
    vs.release()

 
#%%


def test_image(image, detector):
    #image = '../data/person.jpg'
    src_img = cv2.imread(image)
    detector.detect_landmarks(src_img)
    cv2.imshow("Image", src_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



dnn_detector = dnn_detector(detector_conf=0.5, tracker_conf=10)
dlib_detector = detector_dlib()

run_video(args["video"])











