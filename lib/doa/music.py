import numpy as np

from .doa import DOA


class MUSIC(DOA):
    """
    Class to apply MUltiple SIgnal Classication (MUSIC) direction-of-arrival
    (DoA) for a particular microphone array.

    .. note:: Run locate_source() to apply the MUSIC algorithm.

    Parameters
    ----------
    L: numpy array
        Microphone array positions. Each column should correspond to the
        cartesian coordinates of a single microphone.
    fs: float
        Sampling frequency.
    nfft: int
        FFT length.
    c: float
        Speed of sound. Default: 343 m/s
    num_src: int
        Number of sources to detect. Default: 1
    mode: str
        'far' or 'near' for far-field or near-field detection
        respectively. Default: 'far'
    r: numpy array
        Candidate distances from the origin. Default: np.ones(1)
    azimuth: numpy array
        Candidate azimuth angles (in radians) with respect to x-axis.
        Default: np.linspace(-180.,180.,30)*np.pi/180
    colatitude: numpy array
        Candidate elevation angles (in radians) with respect to z-axis.
        Default is x-y plane search: np.pi/2*np.ones(1)
    frequency_normalization: bool
        If True, the MUSIC pseudo-spectra are normalized before averaging across the frequency axis, default:False
    source_noise_thresh: float
        Threshold for automatically identifying the number of sources. Default: 100
    """

    def __init__(
        self,
        L,
        fs,
        nfft,
        c=343.0,
        num_src=1,
        mode="far",
        r=None,
        azimuth=None,
        colatitude=None,
        frequency_normalization=False,
        source_noise_thresh=100,
        output_dir=".",
        **kwargs
    ):

        DOA.__init__(
            self,
            L=L,
            fs=fs,
            nfft=nfft,
            c=c,
            num_src=num_src,
            mode=mode,
            r=r,
            azimuth=azimuth,
            colatitude=colatitude,
            **kwargs
        )

        self.spatial_spectrum = None
        self.frequency_normalization = frequency_normalization
        self.source_noise_thresh = source_noise_thresh
        self.output_dir = output_dir
        self.spectra_storage = []
        self.decomposed_values_strage = []
        self.decomposed_vectors_strage = []

    def _process(self, X, _, auto_identify, **kwargs):
        """
        Perform MUSIC for given frame in order to estimate steered response
        spectrum.
        """
        # compute steered response
        self.spatial_spectrum = np.zeros((self.num_freq, self.grid.n_points))
        R = self._compute_correlation_matricesvec(X)
        # subspace decomposition
        noise_subspace = self._extract_noise_subspace(R, auto_identify=auto_identify)
        # compute spatial spectrum
        self.spatial_spectrum = self._compute_spatial_spectrum(noise_subspace)

        if self.frequency_normalization:
            self._apply_frequency_normalization()
        self.grid.set_values(np.squeeze(np.sum(self.spatial_spectrum, axis=1) / self.num_freq))
        self.spectra_storage.append(self.grid.values)

    def _compute_correlation_matricesvec(self, X):
        # change X such that time frames, frequency microphones is the result
        X = np.transpose(X, axes=[2, 1, 0])
        # select frequency bins
        X = X[..., list(self.freq_bins), :]
        # compute PSD and average over time frame
        C_hat = np.matmul(X[..., None], np.conjugate(X[..., None, :]))
        # average over time-frames
        C_hat = np.mean(C_hat, axis=0)
        return C_hat

    def _extract_noise_subspace(self, R, auto_identify):
        # eigenvalues and eigenvectors are returned in ascending order; no need to sort.
        decomposed_values, decomposed_vectors = np.linalg.eigh(R)

        self.decomposed_values_strage.append(decomposed_values)
        self.decomposed_vectors_strage.append(decomposed_vectors)

        # if auto_identify:
        #     self.num_src = self._auto_identify(decomposed_values)

        noise_subspace = decomposed_vectors[..., :-self.num_src]

        return noise_subspace

    def _plot_decomposed_values(self, decomposed_values):
        import matplotlib.pyplot as plt

        # visualize the magnitude of decomposed values
        fig, axes = plt.subplots(1, 8, figsize=(15, 8), sharey=True)
        for i in range(8):
            axes[i].plot(decomposed_values[..., i], label=f"Value {i+1}", marker="o", linestyle="")
            axes[i].set_title(f"Value {i+1}")
            axes[i].set_xlabel("Row Index")
            axes[i].set_ylim([np.min(decomposed_values), np.max(decomposed_values)])

        axes[0].set_ylabel("Value Magnitude")
        plt.suptitle("Distribution of Decomposed Values Magnitudes Across Rows")

        plt.savefig(f"{self.output_dir}/decomposed_values.png")
        plt.close()

    def _auto_identify(self, decomposed_values):
        """
        Automatically identify the number of sources based on the decomposed values
        of the correlation matrix.
        """
        values_max = np.max(decomposed_values, axis=0)
        # compute the ratio between consecutive decomposed values
        values_ratio = values_max[1:] / values_max[:-1]

        print(f"Decomposed values ratio: {values_ratio}")
        # save the decomposed values ratio
        # np.savetxt(f"{self.output_dir}/decomposed_values_ratio.txt", values_ratio)
        self.dval_ratio_strage.append(values_ratio)

        # find the index where the ratio exceeds the threshold or return the last index
        index = np.argmax(values_ratio > self.source_noise_thresh)
        num_sources = len(values_ratio) - index if index else len(values_ratio)
        return num_sources

    def _compute_spatial_spectrum(self, noise_subspace):
        spatial_spectrum = np.zeros((self.num_freq, self.grid.n_points))
        self.steering_vector = self._compute_steering_vector()
        for f in range(self.num_freq):
            steering_vector_f = self.steering_vector[f, :, :]
            noise_subspace_f = noise_subspace[f, :, :]
            # calculate the MUSIC spectrum for each angle and frequency bin
            vector_norm_squared = np.abs(np.sum(np.conj(steering_vector_f) * steering_vector_f, axis=1))
            noise_projection = np.abs(np.sum(
                np.conj(steering_vector_f) @ noise_subspace_f * (noise_subspace_f.conj().T @ steering_vector_f.T).T,
                axis=1))
            spatial_spectrum[f, :] = vector_norm_squared / noise_projection
        return spatial_spectrum.T

    def _compute_steering_vector(self, angle_step=1):
        n_channels = self.L.shape[1]
        theta = np.deg2rad(np.arange(-180, 180, angle_step))

        direction_vectors = np.zeros((2, len(theta)), dtype=float)
        direction_vectors[0, :] = np.cos(theta)
        direction_vectors[1, :] = np.sin(theta)

        steering_phase = np.einsum("k,ism,ism->ksm", 2.0j * np.pi / self.c * self.freq_hz,
                                   direction_vectors[..., None], self.L[:, None, :])
        steering_vector = 1.0 / np.sqrt(n_channels) * np.exp(steering_phase)
        return steering_vector

    def _compute_spatial_spectrumvec(self, cross):
        mod_vec = np.transpose(
            np.array(self.mode_vec[self.freq_bins, :, :]), axes=[2, 0, 1]
        )
        # timeframe, frequ, no idea
        denom = np.matmul(
            np.conjugate(mod_vec[..., None, :]), np.matmul(cross, mod_vec[..., None])
        )
        return 1.0 / abs(denom[..., 0, 0])

    def _apply_frequency_normalization(self):
        """
        Normalize the MUSIC pseudo-spectrum per frequency bin
        """
        self.spatial_spectrum = self.spatial_spectrum / np.max(self.spatial_spectrum, axis=0, keepdims=True)
