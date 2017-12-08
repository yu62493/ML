####################################################
# Modified by Nazmi Asri                           #
# Original code: http://thecodacus.com/            #
# All right reserved to the respective owner       #
####################################################

import os 

# Import OpenCV2 for image processing
import cv2

# Import numpy for matrices calculations
import numpy as np
from sklearn import preprocessing


class LabelEncoder(object):
    # Method to encode labels from words to numbers
    def encode_labels(self, label_words):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(label_words)

    # Convert input label from word to number
    def word_to_num(self, label_word):
        return int(self.le.transform([label_word])[0])

    # Convert input label from number to word
    def num_to_word(self, label_num):
        return self.le.inverse_transform([label_num])[0]

def get_images_and_labels(input_path):
    label_words = []

    # Iterate through the input path and append files
    for root, dirs, files in os.walk(input_path):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename).replace('\\','/')
            label_words.append(filepath.split('/')[-2]) 
            
    # Initialize variables
    le = LabelEncoder()
    le.encode_labels(label_words)

    return le


if __name__=='__main__':
    
    path_train = 'faces_dataset/train'
    le = get_images_and_labels(path_train)

    # Load prebuilt model for Frontal Face
    cascadePath = "cascade_files/haarcascade_frontalface_default.xml"

    # Create classifier from prebuilt model
    faceCascade = cv2.CascadeClassifier(cascadePath);

    # Create Local Binary Patterns Histograms for face recognization
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Load the trained mode
    recognizer.read('trainner.yml')

    # Set the font style
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize and start the video frame capture
    cam = cv2.VideoCapture(0)

    # Loop
    while True:
        # Read the video frame
        ret, im =cam.read()

        # Convert the captured frame into grayscale
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        # Get all face from the video frame
        faces = faceCascade.detectMultiScale(gray, 1.2,5)

        # For each face in faces
        for(x,y,w,h) in faces:
            # Create rectangle around the face
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

            # Recognize the face belongs to which ID
    #        Id = recognizer.predict(gray[y:y+h,x:x+w])
            predicted_index, conf = recognizer.predict(gray[y:y+h, x:x+w])
            print( str(predicted_index) , "==" , conf )
            predicted_person = le.num_to_word(predicted_index)
            # Check the ID if exist 
            if(conf > 60):
                predicted_person = "unknow"

            # Put text describe who is in the picture
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(im, str(predicted_person), (x,y-40), font, 2, (255,255,255), 3)

        # Display the video frame with the bounded rectangle
        cv2.imshow('im',im) 

        # If 'q' is pressed, close program
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Stop the camera
    cam.release()

    # Close all windows
    cv2.destroyAllWindows()