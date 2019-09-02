import logging
import glob

import cv2

def load_image(image_path):
    """
        Load image as grayscale image
        Args:
            image_path (str): Folder containing images to be loade

        Returns
            image
    """
    # Read Image
    image = cv2.imread(image_path)

    # Convert to grayscale
    image = cvt_to_gray(image)

    return image

def load_images_from_folder(image_folder, max_number=None):
    """
        Load folder of images as grayscale images
        Args:
            image_folder (str): Folder containing images to be processed
            (Optional) max_number (int): maximum number of images to return

        Returns
            images (list): list of imported images
    """
    paths = glob.glob(image_folder + '/*')
    paths = paths[:max_number] if max_number else paths

    imgs = [load_image(p) for p in paths]

    return imgs


def cvt_to_gray(image):
    """
        Converts image to grayscale.
        Args:
            image (HxW): Image to convert


        Returns
            image (HxW np.array): converted image. None if image type cannot be converted
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim != 2:
        logging.warning(f"Image diminsion of {image.ndim} cannot be converted to grayscale")
        return None

    return image


def crop_to_center(image, target_shape):
    """
        Crop an image to a desired width and height while keeping the center

        Args:
            images (np.array): Image to crop
            target_shape (int, int): Height and width of the resulting image

        Returns:
            crop (np.array): Cropped image
    """
    source_h, source_w = image.shape
    target_h, target_w = target_shape

    # Assert target height and width can be achieved using crops
    assert target_h <= source_h,"Crop target height is greater than source image height"
    assert target_w <= source_w,"Crop target width is greater than source image width"

    # Get excess
    top_margin = (source_h - target_h)
    left_margin = (source_w - target_w)

    crop = image[top_margin:top_margin+target_h, left_margin:left_margin+target_w]

    return crop

def resize_image(image, shape, keep_aspect_ratio=True):
    """
        Resize and image to a target shape with or without keeping aspect ratio

        Args:
            images (np.array): Image to crop
            shape (int, int): Height and width of the resulting image
            keep_aspect_ratio (bool): Keep the aspect ratio of original image

        Returns:
            resized (np.array): resize_image
    """
    source_h, source_w = image.shape
    target_h, target_w = shape
