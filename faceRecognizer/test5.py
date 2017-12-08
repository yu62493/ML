import os
import cv2
import numpy as np 
from sklearn import preprocessing

class LabelEncoder(object):
    def encode_labels(self, label_words):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(label_words)

    def word_to_num(self, label_word):
        return int(self.le.transform([label_word])[0])

    def num_to_word(self, label_num):
        return self.le.inverse_transform([label_num])[0]

def get_images_and_labels(input_path):
    label_words = []
    for root, dirs, files in os.walk(input_path):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename).replace('\\','/')
            label_words.append(filepath.split('/')[-2])
            print(filename)
            print(root)
            print(filepath.split('/')[-2])
    
    images = []
    le = LabelEncoder()
    le.encode_labels(label_words)
    labels = []

    for root, dirs, files in os.walk(input_path):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)


if __name__ == '__main__':
    cascade_path = "cascade_files/haarcascade_forntalface_alt.xml"
    path_train = 'faces_dataset/train'
    path_test = 'faces_dataset/test'

    faceCascade = cv2.CascadeClassifier(cascade_path)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images, labels, le = get_images_and_labels(path_train)


