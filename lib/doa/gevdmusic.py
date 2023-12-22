import numpy as np
import scipy

from .music import *
np.set_printoptions(linewidth=150)

class GevdMUSIC(MUSIC):
    """
    Class to apply the Generalized Eigenvalue Decomposition (GEVD) based MUSIC
    (GEVD-MUSIC) direction-of-arrival (DoA) for a particular microphone array,
    extending the capabilities of the original MUSIC algorithm.

    .. note:: Run locate_source() to apply the GEVD-MUSIC algorithm.
    """

    def _process(self, X, X_noise, auto_identify, **kwargs):
        # compute steered response
        self.spatial_spectrum = np.zeros((self.num_freq, self.grid.n_points))
        # compute source and noise correlation matrices
        R = self._compute_correlation_matricesvec(X)
        K = self._compute_correlation_matricesvec(X_noise)
        if kwargs.get("ncm_diff", False):
            K = apply_error_to_hermitian_matrices(K, 0.05)
            for i in range(self.num_freq):
                if not is_positive_definite(K[i]):
                    print("K not positive definite")
        # subspace decomposition
        noise_subspace = self._extract_noise_subspace(R, K, auto_identify=auto_identify)
        # compute spatial spectrum
        self.spatial_spectrum = self._compute_spatial_spectrum(noise_subspace)

        if self.frequency_normalization:
            self._apply_frequency_normalization()
        self.grid.set_values(np.squeeze(np.sum(self.spatial_spectrum, axis=1) / self.num_freq))
        self.spectra_storage.append(self.grid.values)

    def _extract_noise_subspace(self, R, K, auto_identify):
        decomposed_values = np.empty(R.shape[:2], dtype=complex)
        decomposed_vectors = np.empty(R.shape, dtype=complex)

        for i in range(self.num_freq):
            decomposed_values[i], decomposed_vectors[i] = scipy.linalg.eigh(R[i], K[i])
        decomposed_values = np.real(decomposed_values)

        # print(decomposed_values.shape, self.num_freq)
        self.decomposed_values_strage.append(decomposed_values)

        # if auto_identify:
        #     self.num_src = self._auto_identify(decomposed_values)

        noise_subspace = decomposed_vectors[..., :-self.num_src]

        return noise_subspace


def apply_error_to_hermitian_matrices(K, error_ratio):
    """
    Apply random errors to all Hermitian matrices in the given array and
    ensure that all matrices are positive definite.

    :param K: A numpy array of shape (N, M, M) containing N Hermitian matrices.
    :param error_ratio: The ratio of error to be applied.
    :return: A numpy array with the modified Hermitian matrices.
    """
    # Copy the original array to avoid modifying it directly
    modified_K = np.real(K)

    # Function to add random error
    def add_random_error(value, ratio):
        error = np.random.uniform(-ratio, ratio)
        return value * (1 + error)

    # Apply error and ensure positive definiteness
    for matrix in modified_K:
        # Apply error to real part (upper triangular including diagonal)
        for i in range(matrix.shape[0]):
            for j in range(i, matrix.shape[1]):
                matrix[i, j] = add_random_error(matrix[i, j], error_ratio)
                if i != j:
                    matrix[j, i] = matrix[i, j]

        # Ensure positive definiteness
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals[eigvals < 0] = 1e-4
        matrix[:] = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return modified_K


def is_positive_definite(matrix):
        return np.all(np.linalg.eigvalsh(matrix) > 0)
