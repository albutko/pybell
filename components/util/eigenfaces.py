import pickle
import math

import cv2
import numpy as np
from sklearn.decomposition import PCA

def pca(X, nb_components):
    """ Find principal components of a matrix X using SVD

        Args:
            X ((N x H*W) np.array): Matrix of flat image features
            nb_components (int): Number of principal components to return, must be less than feature
                        size of X

        Returns:
            principal_components (nb_components x H*W np.array): principal components
            mean_X (H*W x 1 np.array): mean instance
    """

    instances, n_features = X.shape

    assert nb_components <= n_features, "Number of principal components /\
                    must be less than domain feature space"

    mean_X = np.mean(X, axis=0)

    centered_X = X - mean_X

    u, s, vh = np.linalg.svd(centered_X)
    principal_components = vh[:nb_components]
    return principal_components, mean_X


def extract_facespace(face_imgs, components, project=False):
    """ Create basis for facespace from image of faces using Eigenface pipeline

        Args:
            face_imgs ((N x H x W) np.array): Train set of grayscale face images
            components (int): number of principal components to keep from PCA
            (Optional) project (boolean): return projections of train set or no

        Returns:
            face_space ((components x H*W) np.array): Basis of face space to be used
                    for projection
            mean_face: ((H*W x 1) np.array): Mean face of face images
            (Optional) projections ((N x components) np.array) Projections of face_imgs
                onto face_space if project is True
    """
    n_faces, H, W = face_imgs.shape

    # Flatten Images to feature vector
    face_vectors = face_imgs.reshape((n_faces,-1))

    pca = PCA(n_components=components)

    # Extract Principal Components
    pca.fit(face_vectors)

    facespace = pca.components_
    mean_face = pca.mean_
    # If projections are desired
    if project:
        projection = pca.transform(face_vectors)

        return facespace, mean_face, projection

    return facespace, mean_face

def project(facespace, x):
    """ Project vector on to basis

        Args:
            facespace ((N x H*W) np.array): Matrix representing basis
            x (H*W np.array): vector to project

        Returns:
            projection (N dim np.array): projection of x onto face_space
    """
    projection = np.matmul(facespace, x)
    return projection


def save_facespace_dict(facespace, mean_face, output_file):
    """Save facespace basis and mean_face to dict"""
    facespace_dict = {'facespace': facespace, 'mean_face': mean_face}

    with open(output_file, 'wb') as f:
        pickle.dump(facespace_dict, f)


def load_facespace_dict(input_file):
    """Load facespace basis and mean_face from dict"""
    with open(input_file, 'rb') as f:
        facespace_dict = pickle.load(f)

    assert 'facespace' in facespace_dict, f'FaceSpace not stored in {input_file}'
    assert 'mean_face' in facespace_dict, f'mean_face not stored in {input_file}'

    return facespace_dict


def distance_matrix(vec, vecs):
    """Calculate distance between a vector and all other vectors"""

    result = np.sqrt(np.sum((vecs - vec)**2, axis=1))
    return result
