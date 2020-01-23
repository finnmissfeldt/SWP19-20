import cv2 
import dlib

class detector_dlib:
    """
    Dlib's Gesichterkennung und landmark-prediction.
    """
    
    def __init__(self):    
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("detection/models/shape_predictor_68_face_landmarks.dat")
        self.color = (0, 0, 255)
        
    
    
    def detect_faces(self, image_gray):        
        # Gesichter finden
        faces = self.detector(image_gray)
        #print(str(len(faces_dlib)) + ' Gesichter gefunden')
        for face in faces:
            #Koordinaten
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(image_gray, (x1, y1), (x2, y2), self.color, 1)
        
        return faces
    
            
    def detect_landmarks(self, image):
        # Farbbild in Graubild konvertieren
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(image_gray)
        
        all_landmarks = []
        
        for face in faces:
        # fuer jedes Gesicht landmarks zeichnen
            landmarks = self.predictor(image_gray, face)
            all_landmarks.append(landmarks)
                
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(image, (x, y), 2, self.color, -1)
        
        return all_landmarks

        




