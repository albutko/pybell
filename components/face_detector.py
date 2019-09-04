""" FaceDetector detects faces in grayscale images"""

import os
import logging

import cv2

from .util.image import cvt_to_gray

dirname = os.path.dirname(os.path.abspath(__file__))

class FaceDetector:
    def __init__(self, cascade=os.path.join(dirname, '../data/cascades/haarcascade_frontalface_alt.xml'), face_shape=(64,64)):
        """ Creates a face detection object that finds faces using multiscale Haar cascades.
            In this implementation we use the pretrained Haar Cascade files provided by OpevCV

            Attributes:
                cf (cv2.CascadeClassifier): Cascade Classifier
                face_shape (tuple): Height and Width of resulting faces should be output as

        """
        self.cf = cv2.CascadeClassifier()
        if not self.cf.load(cascade):
            raise FileNotFoundError(f"Cascade file not found at: {cascade}")

        self.face_shape = face_shape

    def detectFaces(self, img):
        """
            Detect faces in an image

            Args:
                img (np.array): Image to detect faces in

            Returns:
                face_count (int): Number of faces detected
                bboxes (List): list of (x, y, width, height) tuples representing bounding boxes of detected faces
                faces (List): list of extracted face image subsets from image
        """
        # Convert to Gray
        img = cvt_to_gray(img)

        # Detect Faces using multiscale detector
        bboxes = self.cf.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        faceCount = len(bboxes)

        # Extract Faces
        faces = _extract_faces_from_bbox(img, bboxes)

        logging.info(f"{faceCount} faces detected")

        return (faceCount, bboxes, faces)


def _extract_faces_from_bbox(img, bboxes):
    """
        Extract regions of an image based on list of bounding boxes

        Args:
            img (np.array): Image from which bounding boxes where detected
            bboxes (List): list of (x, y, width, height) tuples representing bounding boxes

        Returns:
            faces (list): Faces encompassed by bounding boxes
    """
    faces = list()
    for (x, y, w, h) in bboxes:
        faces.append(img[y:y+h,x:x+w])

    return faces
