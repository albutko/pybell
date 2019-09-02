import glob

import numpy as np
import cv2

from .exceptions import ImageLoadException, ChessboardCornerException


def get_calibration_coefficients(image_folder, board_size = (7,7)):
    """ Kind distortion coefficients K, D for fisheye camera model

        Args:
            image_path (str): Path to calibration chessboard image
            board_size (tuple): Chessboard Size

        Returns:
            K (3x3 np.ndarray): Calibration camera matrix
            D (4x1 np.ndarray): distortion coefficients

    """
    N_OK = 0
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    # Read in image input

    img_paths = glob.glob(image_folder+'/*')

    imgPoints = list()
    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            raise ImageLoadException()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find Chessboard Corners
        retval, corners = cv2.findChessboardCorners(img, board_size)

        #Check if corners were found
        if retval:
            N_OK += 1
            cv2.cornerSubPix(img, corners, winSize=(5,5), zeroZone=(-1,-1), criteria=subpix_criteria)

            corners = np.transpose(corners, (1,0,2))
            imgPoints.append(corners)

    objPoints = np.zeros((1, board_size[0]*board_size[1], 3))
    objPoints[0,:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)

    corners = corners[np.newaxis,:,:,:]
    objPoints = objPoints[np.newaxis,:,:,:]

    K = np.zeros((3,3))
    D = np.zeros((1,4))

    cv2.fisheye.calibrate(objPoints, corners, img.shape, K, D)

    return (K, D)
