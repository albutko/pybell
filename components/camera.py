""" Camera module used for input processing """
import os

import numpy as np
import cv2

import components.util.camera as util

class CameraStream:
    """ Class creates a camera object used for retrieving frames for facial recognition
        processing

        As this project is meant for use in apartment door peepholes, this camera class
        is able to hande fisheye distortion on images

        Attributes:
            video_capture: OpenCV video capture object using specificed camera object
            undistort_fisheye: A boolean indicating whether to undistort output images
            K: (3x3) np.array camera calibration matrix used for undistorting fisheye
            D: (1x4) np.array distortion coefficients used for undistorting fisheye
            calibration_file: pickle file with saved calibration information, to be used
                if K, D are not specified

    """

    def __init__(self, camera=0, undistort_fisheye = False, K=None, D=None, calibration_file=None):
        """Initialize Camera Stream Object"""
        self.video_capture = cv2.VideoCapture(camera)
        self.undistort_fisheye = undistort_fisheye
        self.K = K
        self.D = D
        if calibration_file:
            self._set_calibration_values(calibration_file)


    def stop(self):
        """Stop the video capture"""
        self.video_capture.release()

    def next_frame(self):
        """Return the next frame of the video capture, undistorting if undistort_fisheye"""
        _, img = self.video_capture.read()

        if self.undistort_fisheye:
            img = cv2.fisheye.undistortImage(img, self.K, self.D)

        return img

    def _set_calibration_values(self, calib_file):
        """Set calibration values from calibration dictionary file"""
        calib_dict = util.load_calibration_coefficients(calib_file)

        self.K = calib_dict.get('K')
        self.D = calib_dict.get('D')
