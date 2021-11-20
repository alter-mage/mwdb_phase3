from sklearn.cluster import KMeans
import numpy as np
import scipy
import scipy.spatial

import min_max_scaler


class k_means:
    """
    Represents K-means feature reduction class class
    ...
    Attributes:
        k: int
            Number of reduced features

        X: ndarray of shape (num_objects, num_features)
            Data matrix to be reduced

    Methods:
        get_latent_semantics()
            Returns k reduced latent semantics of X

        transform(X)
            Transforms and returns X in the latent semantics space
    """
    def __init__(self, k, X):
        """
        Parameters:
            k: int
                Number of reduced features

            X: ndarray of shape (num_objects, num_features)
                Data matrix to be reduced
        """
        self.x_ = np.array(X, dtype=np.float32)
        self.k_means_ = KMeans(
            n_clusters=k
        ).fit(self.x_)

        self.right_fac_ = self.k_means_.cluster_centers_
        self.x_t_ = self.x_.transpose()
        self.left_fac_ = np.reciprocal(
            np.array(scipy.spatial.distance.cdist(self.x_, self.right_fac_, metric='euclidean'), dtype=np.float32)
        )
        self.centre_mat_ = []

    # def get_left_factor_matrix(self):
    #     """
    #     Returns:
    #         k latent semantic features
    #     """
    #
    #
    #     return self.left_fac_, self.centre_mat_, self.right_fac_

    def transform(self):
        """
        Parameters:
            X: ndarray of shape (num_objects, num_features)
                Data matrix to be reduced

        Returns:
            Transforms and returns X in the latent semantics space
        """
        return self.left_fac_, self.centre_mat_, self.right_fac_


def get_transformation(data, right_matrix):
    if len(data.shape) != 2:
        data = np.reshape(data, (1, len(data)))
    return np.reciprocal(
        np.array(scipy.spatial.distance.cdist(np.array(data), np.array(right_matrix), metric='euclidean'), dtype=np.float32)
    )


# Keeping this alive in case, definitely not using this rn!
# def compute_k_means(X):
#     kmeans = KMeans(n_clusters=8,
#                     init='k-means++',
#                     n_init=10,
#                     max_iter=300,
#                     algorithm='auto',
#                     random_state=0).fit(X)
#     return kmeans


# Note for other team members: I have kept the default values for all parameters for K-Means right now but we can change it as and when needed. If the k-means parameters are different for each task then I will change the function to keep all the values as parameters in the function itself.