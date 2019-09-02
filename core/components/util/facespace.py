

import cv2
import numpy as np

def pca(X, nb_components):
    """ Find principal components of a matrix X using SVD

        Args:
            X ((N x H*W) np.array): Matrix of flat image features
            nb_components (int): Number of principal components to return, must be less than feature
                        size of X


        Returns:
            face_space ((components x H*W) np.array): Basis of face space to be used
                    for projection
            (Optional) projections ((N x components) np.array) Projections of face_imgs
                if project is True
    """

    instances, n_features = X.shape

    assert nb_components <= n_features, "Number of principal components /\
                    must be less than domain feature space"

    mean_X = np.mean(X, axis=0)

    centered_X = X - mean_X

    u, s, vh = np.linalg.svd(centered_X)
    principal_components = vh[:nb_components]
    return principal_components



def extract_facespace(face_imgs, components, project=False):
    """ Create basis for facespace from image of faces using Eigenface pipeline

        Args:
            face_imgs ((N x H x W) np.array): Train set of grayscale face images
            components (int): number of principal components to keep from PCA
            (Optional) project (boolean): return projections of train set or no

        Returns:
            face_space ((components x H*W) np.array): Basis of face space to be used
                    for projection
            (Optional) projections ((N x components) np.array) Projections of face_imgs
                onto face_space if project is True
    """
    n_faces, H, W = face_imgs.shape

    # Flatten Images to feature vector
    face_vectors = face_imgs.reshape((n_faces,-1))

    # Extract Principal Components
    face_space = pca(face_vectors, components)

    # If projections are desired
    if project:
        mean_face = np.mean(face_vectors, axis=0)
        mean_centered = face_vectors - mean_face

        projections = np.matmul(face_space, mean_centered.T)

        return face_space, projections

    return face_space
