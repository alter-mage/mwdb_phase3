# coding=utf-8
import numpy as np


# it's not realizing the attributes, not sure why mine is having issues, might just be my IDE


class svd:
    """
        Represents SVD feature reduction class
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
                Transforms and returns X in the latent semantic space and latent semantics
        """

    def __init__(self, k, data_matrix):
        """
        Parameters:
            Datamatrix: ndarray of shape (num_objects, num_features)
                Data matrix to be reduced
            k: int
                Number of reduced features
        """

        self.matrix1 = data_matrix @ data_matrix.transpose()
        self.matrix2 = data_matrix.transpose() @ data_matrix

        self.eigen_values1, self.eigen_vectors1 = np.linalg.eigh(self.matrix1)
        self.eigen_values2, self.eigen_vectors2 = np.linalg.eigh(self.matrix2)

        self.eigen_values1, self.eigen_values2 = self.eigen_values1[::-1], self.eigen_values2[::-1]
        self.eigen_vectors1, self.eigen_vectors2 = self.eigen_vectors1[:, ::-1], self.eigen_vectors2[:, ::-1]

        if len(self.eigen_values1) > len(self.eigen_values2):
            self.c_ = self.eigen_values1
        else:
            self.c_ = self.eigen_values2
        self.c_ = self.c_[:min(len(self.eigen_values1), len(self.eigen_values2))]
        self.c_ = np.diag(np.sqrt(self.c_)[:k])

        self.left_ = self.eigen_vectors1[:, :k]
        self.right_ = self.eigen_vectors2[:, :k]

    def transform(self):
        """
        parameters:
            X: The matrix of object*features
            k: Number of latent features

        returns:
            Matrix of K latent features and latent semantics
        """

        # might want fit_transform, but should have been fitted already so ¯\_(ツ)_/¯
        return self.left_, self.c_, self.right_


def get_transformation(data, right_matrix):
    return np.dot(np.array(data), np.array(right_matrix))

    # Might be helpful later
    # def compute_svd_reverse(X, k):
    #    trun_svd = TruncatedSVD(n_components=k)
    #    X_original = trun_svd.fit_transform(X)
    #    return X_original

    # components_ndarray of shape (n_components, n_features)
    # The right singular vectors of the input data.

    # explained_variance_ndarray of shape (n_components,)
    # The variance of the training samples transformed by a projection to each component.

    # explained_variance_ratio_ndarray of shape (n_components,)
    # Percentage of variance explained by each of the selected components.

    # singular_values_ndarray od shape (n_components,)
    # The singular values corresponding to each of the selected components. The singular values are equal to the 2-norms of the n_components variables in the lower-dimensional space.

    # n_features_in_int
    # Number of features seen during fit.
