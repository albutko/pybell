import numpy as np
import cv2

class CameraStream:
    def __init__(self, config):
        self.video_capture = cv2.VideoCapture(config.device)

    def start(self):
        self.video_capture.open()

    def stop(self):
        self.video_capture.release()

    def next_frame(self):
        _, img = self.video_capture.read()
        return img

    def _get_distortion_coefficients(self):
        pass
