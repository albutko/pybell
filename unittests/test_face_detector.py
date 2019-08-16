import unittest

import glob
import cv2

import core.utils as utils
from core.face_detector import FaceDetector, draw_detections
from core.exceptions import ImageLoadException, ChessboardCornerException

class TestFaceDetector(unittest.TestCase):
    def test_can_loads_cascade(self):
        cascade_paths = glob.glob('./core/cascades/*')
        cf = cv2.CascadeClassifier()
        cf.load(cascade_paths[0])
        self.assertTrue(cf.load(cascade_paths[0]))

    def test_cascade_file_doesnot_exists_throws_error(self):
        with self.assertRaises(FileNotFoundError):
            detector = FaceDetector('../cascade_fileDNE.xml')


    def test_face_detector_singleface_face_count_is_one(self):
        detector = FaceDetector()
        img = cv2.imread('./unittests/test_images/singleface.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceCount, _ = detector.detectFaces(img)
        self.assertEqual(faceCount, 1)

    def test_face_detector_singleface_bbox_shape_one_by_four(self):
        detector = FaceDetector()
        img = cv2.imread('./unittests/test_images/singleface.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _ , bboxes = detector.detectFaces(img)
        self.assertEqual(bboxes.shape, (1,4))

    def test_face_detector_multiface_face_count_is_four(self):
        detector = FaceDetector()
        img = cv2.imread('./unittests/test_images/multiface.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceCount, _ = detector.detectFaces(img)
        self.assertEqual(faceCount, 3)

    def test_face_detector_multiface_bboxs_is_four_by_four(self):
        detector = FaceDetector()
        img = cv2.imread('./unittests/test_images/multiface.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _ , bboxes = detector.detectFaces(img)
        self.assertEqual(bboxes.shape, (3,4))

    def test_face_detector_noface_face_count_is_0(self):
        detector = FaceDetector()
        img = cv2.imread('./unittests/test_images/no_face.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceCount, _ = detector.detectFaces(img)
        self.assertEqual(faceCount, 0)

    def test_face_detector_noface_bboxes_is_None(self):
        detector = FaceDetector()
        img = cv2.imread('./unittests/test_images/no_face.png')
        _ , bboxes = detector.detectFaces(img)
        self.assertIsNone(bboxes)

if __name__ == "__main__":
    unittest.main()
