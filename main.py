import cv2
import os
from detection.dnn_detector import dnn_detector
from detection.dlib_detector import detector_dlib
import argparse
import numpy as np
from face_swap import warp_image_3d, mask_from_points, apply_mask, correct_colours
from detection.face_points_detection import face_points_detection

"""
Verarbeitet das uebergene Video Frame fuer Frame indem Gesichter erkannt 
und durch Bilder aus einem übergebenen Ordner ersetzt werden.
Zu Gesichtern die mit dem Dlib-frontal-face-detector erkannt werden, werden
68-markente Punkte des Gesichtes angezeigt.
Separat werden durch ein vortrainiertes KI-Model von OpenCV weitere Gesichter erkannt,
da Dlib leider nicht alle erkennt.
"""

# Parameter der Kommandozeile parsen
parser = argparse.ArgumentParser(description='Gesichtsersetzungstool')
parser.add_argument('--src_path', help='Pfad für einzusetzende Bilder (Ordner)')
parser.add_argument('--dst_video', help='Pfad für zu ersetzendes Video')
parser.add_argument('--out', help='Pfad für ersetzes Video')
args = parser.parse_args()

    

#%%
def run_video(video, video_out, src_path):
    
    #liest das Video ein
    cap = cv2.VideoCapture(video)
    [grabbed, frame] = cap.read() 
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    #VideoWriter, um ein Video in dem übergebenen Pfad zu speichern
    out = cv2.VideoWriter(video_out ,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
    # Iteration ueber die Frames des Videos
    while True:
    	# naechstes Frame auslesen
        [grabbed, frame] = cap.read()
    
    	# Ende des Videos abfangen
        if frame is None:
            break
        
        # Gesichter erkennen mit dem DNN-Modul (gruene Rechtecke)
        dst_faces = dnn_detector.detect_faces(frame)
        # error if no faces detected
        if len(dst_faces) == 0:
            print('Keine Gesichter in dst_image gefunden!')
            exit(-1)
        
        #Kopie des Bildes, da es verändert wird
        dst_img_cp = frame.copy()
        
        counter = 0
        #Iteriert über die gefundene Gesichter
        for face in dst_faces:
            #liest das erste Bild aus dem Ordner ein
            src_img = cv2.imread(src_path + '/' + imgs[counter])
            #erkennt die Gesicht in dem entsprechendem Bild
            src_faces = dnn_detector.detect_faces(src_img)
            
            # error if no faces detected
            if len(src_faces) == 0:
                print('Keine Gesichter in src_img gefunden!')
                exit(-1)
            src_face = src_faces[0]
            
            #liest die Landmarks
            src_points, src_shape, src_face = select_face(src_img, src_face)
            
            # Wählt das Zielgesicht aus
            dst_points, dst_shape, dst_face = select_face(frame, face)
            
            h, w = dst_face.shape[:2]
            warped_src_face = warp_image_3d(src_face, src_points[:48], dst_points[:48], (h, w))
            ## Überlagert die Gesichter
            mask = mask_from_points((h, w), dst_points)
            mask_src = np.mean(warped_src_face, axis=2) > 0
            mask = np.asarray(mask*mask_src, dtype=np.uint8)
            
            # Korrigiert die Farben
            warped_src_face = apply_mask(warped_src_face, mask)
            dst_face_masked = apply_mask(dst_face, mask)
            warped_src_face = correct_colours(dst_face_masked, warped_src_face, dst_points)
            
            ## Verkleinert das Gesicht
            kernel = np.ones((10, 10), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            r = cv2.boundingRect(mask)
            center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
            output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)
            
            x, y, w, h = dst_shape
            dst_img_cp[y:y+h, x:x+w] = output
            output = dst_img_cp
            if (counter + 1 < len(imgs)):
                counter+=1
            else:
                counter = 0
        
        # Zeigt das resultierende Video an
        out.write(output)
        cv2.imshow('frame',output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
    

    
    # aufraeumen
    cv2.destroyAllWindows()
    cap.release()
    out.release()

 
#%%
    
#Liest Bilder aus dem übergebenen Pfad aus
def list_all(path):
    """
    Listet Dateien und Verzeichnisse in diesem ('.') auf
    """
    imgs = []
    for folder in os.listdir(path):
        imgs.append(folder)
    return imgs

#Gibt die Landmarks für das entsprechende Gesicht zurück
def select_face(im, bbox, r=10):
    points = np.asarray(face_points_detection(im, bbox))
    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)
    
    x, y = max(0, left-r), max(0, top-r)
    w, h = min(right+r, im_h)-x, min(bottom+r, im_w)-y
    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y+h, x:x+w]


#Initialisierung von dnn_detector und dlib_detector
dnn_detector = dnn_detector(detector_conf=0.5, tracker_conf=10)
dlib_detector = detector_dlib()

#liest alle Bilder aus dem übergebenen Ordner aus
imgs = list_all(args.src_path)
#bearbeitet das Video und ersetzt alle erkannten Gesichter
run_video(args.dst_video, args.out, args.src_path)











