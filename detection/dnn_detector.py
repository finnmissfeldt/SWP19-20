import cv2 
import numpy as np
import dlib

class dnn_detector:
    """
    DNN Face Detector in OpenCV.
    Das Model wird von OpenCV bereitgestellt.
    Es wird versucht in jedem Frame alle Gesichter zu finden und anzuzeigen.
    Zusätzlich wird der Correlation-Tracker von Dlib jedes Gesicht getrackt,
    um in Frames in denen das Gesicht nicht erkannt wird verfolgen zu können.
    
    """
    
    def __init__(self, detector_conf, tracker_conf):
        modelFile = 'detection/models/res10_300x300_ssd_iter_140000.caffemodel'
        configFile = 'detection/models/deploy.prototxt.txt'
        self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        self.predictor = dlib.shape_predictor("detection/models/shape_predictor_68_face_landmarks.dat")
        self.trackers = []
        self.conf_threshold = detector_conf
        self.conf_track = tracker_conf
        self.color = (0, 255, 0)
    
    
    def detect_faces(self, image, saveImg, newVersion):
        faces = []
        coordinates = []
        
        # Bild wird in ein Blob konvertiert und durch das Netzwerk verarbeitet
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        (h, w) = image.shape[:2]
        
        # Iteration ueber alle Treffer
        for i in range(0, detections.shape[2]):
            
            # Wahrscheinlichkeit pruefen
            confidence = detections[0, 0, i, 2]
         
            if confidence > self.conf_threshold:
        		# (x, y)-Koordinaten extrahieren
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    
                (startX, startY, endX, endY) = box.astype('long')
                
                # fuer dlib landmark-detection speichern
                faces.append(dlib.rectangle(startX, startY, endX, endY))
                
                # Mittelpunkt des Gesichtes um vorhandene Tracker zu pruefen
                detection_center_x = startX + 0.5 * (endX - startX)
                detection_center_y = startY + 0.5 * (endY - startY)
                        
                matchID = None                  
                        
                for tracker in self.trackers:
                    position = tracker.get_position()
                                
                    t_x = int(position.left())
                    t_y = int(position.top())
                    t_w = int(position.width())
                    t_h = int(position.height())
                                
                    #Mittelpunkt
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h
                                 
                    if ( ( t_x <= detection_center_x   <= (t_x + t_w)) and 
                        ( t_y <= detection_center_y   <= (t_y + t_h)) and 
                        ( startX   <= t_x_bar <= (startX   + (endX - startX)  )) and 
                        ( startY   <= t_y_bar <= (startY   + (endY - startY)  ))): 
                        matchID = tracker
                                     
                if matchID is None:
                    self.create_tracker(startX, startY, endX, endY, image)
                    
        self.update_trackers(image)
        if (saveImg == True) :
            self.draw_rects(image)
        
           
        if (not newVersion):
            return faces
        else:
            for t in self.trackers:
                pos = t.get_position()
                    
                # Koordinaten des Rechtecks eines Trackers 
                startX = int(pos.left() - 20)
                startY = int(pos.top() - 20)
                endX = int(pos.right() + 20)
                endY = int(pos.bottom() + 20)
                
                coordinates.append([startY,endY, startX, endX])
            return coordinates

                  
    def detect_landmarks(self, image):
        # Farbbild in Graubild konvertieren
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(image_gray)
        
        for face in faces:
        # fuer jedes Gesicht landmarks zeichnen
            landmarks = self.predictor(image_gray, face)
                
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(image, (x, y), 1, self.color, -1)
                
                
    def create_tracker(self, startX, startY, endX, endY, image):
        #print("neuer Tracker wird erstellt")
        				
        # neuen Tracker initialisieren und starten
        t = dlib.correlation_tracker()
        rect = dlib.rectangle(startX, startY, endX, endY)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        t.start_track(rgb, rect)
                
        self.trackers.append(t)
        
    
    def update_trackers(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        delete = []
        for fid in self.trackers:
            quality = fid.update(rgb)
                    
            if quality < self.conf_track :
                delete.append(fid)
                        
        for fid in delete:
            self.trackers.remove(fid)
            
            

    def draw_rects(self, frame):
    
        counter = 0
        for t in self.trackers:
            pos = t.get_position()
                
            # Koordinaten des Rechtecks eines Trackers 
            startX = int(pos.left() - 20)
            startY = int(pos.top() - 20)
            endX = int(pos.right() + 20)
            endY = int(pos.bottom() + 20)
            
            croping = frame[startY:endY, startX:endX]
            name = "tmp/test" + str(counter) + ".jpg"
            cv2.imwrite(name, croping)
                
            # Rechteck des Trackers zeichnen            
            cv2.rectangle(frame, (startX, startY), (endX, endY),
            self.color, 1)
            counter += 1  
