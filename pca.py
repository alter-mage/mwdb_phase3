from sklearn.decomposition import PCA
import numpy as np


class pca:
    """
    Represents PCA dimension technique
    ...
    Attributes:
        k: int
            Number of reduced features

        X: ndarray of shape (num_objects, num_features)
            Data matrix to be reduced

    Methods:
        transform(X)
            Transforms and returns X in the latent semantic space and the latent semantics
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
        self.features_ = self.x_.shape[1]

        self.x_covariance_ = np.cov(self.x_.transpose())
        self.eigen_values_, self.eigen_vectors_ = np.linalg.eigh(self.x_covariance_)
        self.eigen_values_ = self.eigen_values_[::-1]
        self.eigen_vectors_ = self.eigen_vectors_[ : , ::-1]

        self.left_, self.c_, self.right_ = np.dot(self.x_, self.eigen_vectors_[:][:k].transpose()), \
                                              np.diag(self.eigen_values_[:k]), \
                                              self.eigen_vectors_[:, :k]

    def transform(self):
        """
        Parameters:
            X: ndarray of shape (num_objects, num_features)
                Data matrix to be reduced

        Returns:
            Transforms and returns X in the latent semantic space and the latent semantics
        """
        return self.left_, self.c_, self.right_


def get_transformation(data, right_vector):
    return np.dot(np.array(data), np.array(right_vector))


if __name__ == '__main__':
    dummy_data = [[1, 2, 3], [2, 4, 6]]
    pca_obj = pca(1, dummy_data)
    pca_obj.transform()
