import cv2
import numpy as np

from util.eigenfaces import create_facespace, project, distance_matrix
from util.image import load_images_from_folder

class FaceRecognizer:
    def __init__(self, mean_face, facespace, known_face_dict, img_shape=(64,64), thresh=0.1):
        """An object able to recognize input faces using the EigenValue Procedure

           Args:
                mean_face (np.array): (1x(H*W)) mean face of training data
                facespace (np.array): (Nx(H*w)) projection matrix from image to facespace
                                      using N components
                known_face_dict (dict): (Name (str), faceImage) key-value pairs linking persons'
                                        names to their face image!
                thresh (float): threshold for eigenface recognition
        """
        self.mean_face = mean_face
        self.facespace = facespace
        self.known_face_dict = known_face_dict
        self.thresh = thresh
        self.img_shape = img_shape
        self.index2name = list()
        self.known_face_matrix = list()

        for key, faceImg in known_face_dict.values():
            self.index2name.append(key)
            self.known_face_matrix.append(self._project_to_facespace(faceImg))

        self.known_face_matrix = np.array(self.known_face_matrix)


    def _set_facespace(self, facespace):
        """Set FaceScape Attribute"""
        self.facespace = facespace

    def add_known_face(self, name, faceImg):
        """Add face to known face dictionary"""
        self.known_face_dict[name] = faceImg
        self.index2name.append(name)
        projection = project(self.facespace, faceImg.flatten())
        self.known_face_matrix = np.vstack((self.known_face_matrix, projection))

    def recognize(self, faceImg):
        """Check whether a face is recognized from the set of known faces"""
        projection = self._project_to_facespace(faceImg)

        # Find distance between projection and known faces
        dist = distance_matrix(self.known_face_matrix, projection)
        min_idx = np.argmin(dist)

        name = None

        if dist[min_idx] <= self.thresh:
            name = self.index2name[min_idx]
        return name

    def _project_to_facespace(self, faceImg):
        """Project Image of a face to the face recognizer's face space"""
        # Mean center face
        mean_centered = faceImg.flatten() - self.mean_face

        # Project to facespace
        projection = project(self.facespace, mean_centered)

        return projection
