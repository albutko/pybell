#! user/bin/env python
"""Main file used to run facial recognition and detection

Usage: python pybell.py

"""

import cv2

from components.camera import CameraStream
from components.face_detector import FaceDetector
from components.face_recognizer import FaceRecognizer
from components.util.eigenfaces import load_facespace_dict
from components.util.detection import draw_detection_with_label


if __name__ == "__main__":
    # Initialize FaceDetector
    face_detector = FaceDetector()

    # Initialize FaceRecognizer
    face_space_dict = load_facespace_dict('./data/faces/facespace.pkl')
    facespace = face_space_dict['facespace']
    mean_face = face_space_dict['mean_face']

    # Initialize empty known face dictionary
    known_face_dict = dict()
    face_recognizer = FaceRecognizer(mean_face, facespace, known_face_dict, thresh=6000, contrast_coeff=1, brightness_coeff=100)

    # Open and Start Camera
    cam = CameraStream()

    # Continuous stream of input
    while True:
        # Get next frame
        current_frame = cam.next_frame()

        # Detect faces in current frame
        nb_faces, bboxes, faces = face_detector.detectFaces(current_frame)

        # Set unknown flag to false
        unknown = False

        # Iterate through Detections
        for bbox, face in zip(bboxes, faces):
            resized = cv2.resize(face, (64,64))
            name = face_recognizer.recognize(resized)

            # If face was recogized
            if name:
                print(f'{name} found')
                # Draw Frame of found face
                draw_detection_with_label(current_frame, bbox, name, known=True)
            # If unknown
            else:
                draw_detection_with_label(current_frame, bbox, 'UNKNOWN', known=False)
                unknown = True

        # Show Current frame with detections and labels
        cv2.imshow('Face Detections', current_frame)
        cv2.waitKey(1)

        # Allow user to add uknown face as known
        if unknown:
            ans = input('Add face to list of known faces? [y/n]: ')
            if ans[0].lower() == 'y':
                name = input('Input Name: ')
                face_recognizer.add_known_face(name, resized)

        cv2.destroyAllWindows()
