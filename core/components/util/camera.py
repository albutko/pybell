import pickle

import numpy as np
import cv2


from .exceptions import ImageLoadException, ChessboardCornerException
from image import cvt_to_gray

def calculate_calibration_coefficients(images, board_size = (7,7)):
    """ Kind distortion coefficients K, D for fisheye camera model

        Args:
            images (List): List of images with Chessboard Calibration pattern
            board_size (tuple): Chessboard Size

        Returns:
            K (3x3 np.ndarray): Calibration camera matrix
            D (4x1 np.ndarray): distortion coefficients

    """
    N_OK = 0
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    # Read in image input

    imgPoints = list()
    for img in images:

        img = cvt_to_gray(img)

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


def save_calibration_coefficients(K, D, file_path):
    """ Save calibration matrix, K, and distortion coefficients, D, to file as a dict
        {
            K: np.array,
            D: np.array
        }

        Args:
            K (3x3 np.ndarray): Calibration camera matrix
            D (4x1 np.ndarray): distortion coefficients
            file_path (str): path to output file
    """
    calib_dict = {'K': K, 'D': D}

    with open(filename, 'wb') as f:
        pickle.dump(calib_dict, f)


def load_calibration_coefficients(file_path):
    """ Load calibration matrix, K, and distortion coefficients, D, from file as dict

        Args:
            file_path (str): path to calib file

        Returns:
            calib (dict): Dictionary of necessary calibation coefficients
                {
                    K: np.array,
                    D: np.array
                }
    """
    with open(file_path, 'rb') as f:
        calib = pickle.load(f)

    # Assertions - Contains needed calibration variables
    assert 'K' in calib, f'Calibration matrix not stored in {file_path}'
    assert 'D' in calib, f'Distorition coefficients not stored in {file_path}'

    # Variables are the correct shape
    assert calib.get('K').shape == (3, 3), f'Calibration Matrix is wrong shape {calib.get('K').shape} != (3,3)'
    assert calib.get('D').shape == (1, 4), f'Distortion coefficients is wrong shape {calib.get('D').shape} != (1,4)'

    return calib
