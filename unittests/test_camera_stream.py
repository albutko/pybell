import unittest

from core.camerastream import CameraStream
class TestCameraStreamClass(unittest.TestCase):
    def setUp(self):
        self.camera_stream = CameraStream()

    def test_camera_stream_get_frame(self):
        self.setUp()
        self.assertIsInstance(self.camera_stream.get_frame(), 'numpy.ndarray')

if __name__ == "__main__":
    unittest.main()
