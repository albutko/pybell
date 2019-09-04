import cv2
import numpy as np

from .util.eigenfaces import extract_facespace, project, distance_matrix
from .util.image import load_images_from_folder

class FaceRecognizer:
    def __init__(self, mean_face, facespace, known_face_dict=None, img_shape=(64,64),
                 thresh=0.1, contrast_coeff=1.0, brightness_coeff=50):
        """Facial Recognition object finding similar faces using EigenFaces

           As the visual environment can be different than that of the facespace train set,
           contrast and brightness transforms help to mitigate differences
           Args:
                mean_face (np.array): (1x(H*W)) mean face of training data
                facespace (np.array): (Nx(H*w)) projection matrix from image to facespace
                                      using N components
                known_face_dict (dict): (Name (str), faceImage) key-value pairs linking persons'
                                        names to their face image!
                thresh (float): threshold for eigenface recognition
                contrast_coeff (float): coefficient used to change contrast of input face image
                brightness_coeff (float): coefficient used to change brightness of input face image
        """
        self.mean_face = mean_face
        self.facespace = facespace
        self.known_face_dict = known_face_dict
        self.thresh = thresh
        self.img_shape = img_shape
        self.contrast_coeff = contrast_coeff
        self.brightness_coeff = brightness_coeff
        self.index2name = list()
        self.known_face_matrix = list()

        # If face dictionaryy is non-empty
        if known_face_dict:
            for key, faceImg in known_face_dict.values():
                self.index2name.append(key)
                self.known_face_matrix.append(self._project_to_facespace(faceImg))

        self.known_face_matrix = np.array(self.known_face_matrix)

    def _set_facespace(self, facespace):
        """Set FaceScape Attribute"""
        self.facespace = facespace

    def add_known_face(self, name, faceImg):
        """Add face to known face dictionary"""
        faceImg = self._adjust_brightness_contrast(faceImg)
        self.known_face_dict[name] = faceImg
        self.index2name.append(name)
        projection = project(self.facespace, faceImg.flatten())

        # Handle if this is the first known image
        if len(self.known_face_dict) == 1:
            self.known_face_matrix = projection.reshape(1, -1)
        else:
            self.known_face_matrix = np.vstack((self.known_face_matrix, projection.reshape(1, -1)))



    def recognize(self, faceImg):
        """Check whether a face is recognized from the set of known faces.
            If face is unknown return none.
        """
        faceImg = self._adjust_brightness_contrast(faceImg)
        projection = self._project_to_facespace(faceImg)

        # Find distance between projection and known faces
        if self.known_face_matrix.shape[0] > 0:
            dist = distance_matrix(self.known_face_matrix, projection)
            min_idx = np.argmin(dist)
            print(dist)
            name = None
            if dist[min_idx] <= self.thresh:
                name = self.index2name[min_idx]
            return name

        else:
            return None

    def _project_to_facespace(self, faceImg):
        """Project Image of a face to the face recognizer's face space"""
        # Mean center face
        faceImg = self._adjust_brightness_contrast(faceImg)
        mean_centered = faceImg.flatten() - self.mean_face
        # Project to facespace
        projection = project(self.facespace, mean_centered)
        return projection

    def _adjust_brightness_contrast(self, faceImg):
        """Adjust image contrast and brightness based provided coefficients"""
        # contrast*Image + brightness
        adjusted = self.contrast_coeff*faceImg + np.full_like(faceImg, self.brightness_coeff)
        return adjusted
