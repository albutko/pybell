""" Helper file create facespace from folder of face images """
import argparse

import numpy as np
import cv2

from components.util.eigenfaces import extract_facespace, save_facespace_dict
from components.util.image import load_images_from_folder

parser = argparse.ArgumentParser(description="Extract FaceSpace from folder of"
                                 "Face Images of the same dimensions")

parser.add_argument("--folder", default='./data/training/faces', help="Folder holding face images")
parser.add_argument("--max-images", type=int, default=1000, dest="nb_images", help="Maximum number of images to use")
parser.add_argument("--components", type=int, default=250, dest="nb_components", help="Numbed of Principal Components to keep for face space")
parser.add_argument("--out", default='./data/faces/facespace.pkl', help="Output pickle file for facespace information")

if __name__ == '__main__':
    # Parse Arguments
    args = parser.parse_args()

    # Load Images
    face_imgs = load_images_from_folder(args.folder, max_number=args.nb_images)
    assert len(face_imgs) > 0, f"No faces found in {args.folder}"
    #Extract facespace
    face_space, mean_face = extract_facespace(np.array(face_imgs), args.nb_components, project=False)

    # Save facespace to dictionary for future use
    save_facespace_dict(face_space, mean_face, args.out)
