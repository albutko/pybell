import cv2
import numpy as np

from util.eigenfaces import create_facespace
from util.image import load_images_from_folder

class FaceRecognizer:
    def __init__(self, face_basis, nb_components, face_known_folder, face_unknown_folder):
        self.nb_components = nb_components
        self.face_known_folder = face_known_folder
        self.face_unknown_folder = face_unknown_folder

    def _set_facespace(self):
        imgs = np.array(imgList)

        self.face_space = create_facespace(imgs, self.nb_components)


    def _load_known_faces(self):
        """Loads known face projections from pickle file or known face folder"""
        
        imgs = np.array(imgList)
