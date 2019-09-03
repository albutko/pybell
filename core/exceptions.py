
class ImageLoadException(Exception):
    """ Exception indicating failure of an image load due to a file
        not existing, format being incorrect, etc."""
    def __init__(self):
        pass

class ChessboardCornerException(Exception):
    """ Exception indicating chessboard corners were not correctly found """
    def __init__(self):
        pass
