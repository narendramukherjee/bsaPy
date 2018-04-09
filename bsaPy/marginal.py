import warnings
import numpy as np


def sin(freq, t):
    return np.sin(2*np.pi*freq*t)


def cos(freq, t):
    return np.cos(2*np.pi*freq*t)


class GeneralMarginal:
    """
    Posterior inference over general functions of frequency.
    Read Bretthorst 1988, Chapter 3
    """
    def __init__(self, model_funcs):
        """
        Set up the general model for inference of the marginal posterior probabilities of frequencies

        :param model_funcs: Iterable (tuple, list, ..) of model functions
        """

        # Convert model_funcs to a list if not iterable
        if not iter(model_funcs):
            model_funcs = list(model_funcs)
        # Check if all items of model_funcs are functions
        for func in model_funcs:
            if not callable(func):
                raise TypeError("Not all functions passed: model_funcs")
        self.model_funcs = model_funcs

    def evaluate_model_funcs(self, freq):
        """
        Evaluate the model functions at each time point
        :param freq: A single frequency (in Hz) from the input frequencies
        :return: (len(model_funcs), len(t)) array of each model function evaluated at each time point
        """
        return np.array([func(freq, self.t) for func in self.model_funcs])

    def get_model_matrix(self, freq):
        """
        Evaluate the matrix G with :math:`g_{jk} = \sum\limits_{t=1}^{t=T}G_{j}(t)G_{k}(t)`
        where :math:`G_{j}` and :math:`G_{k}` refer to the jth and kth model function
        :param freq: A single frequency (in Hz) from the input frequencies
        :return: (len(model_funcs), len(model_funcs)) array with elements :math:`g_{jk}`
        """
        G_t = self.evaluate_model_funcs(freq)
        G = np.array([[np.sum(G_t[i, :]*G_t[j, :]) for j in range(G_t.shape[0])] for i in range(G_t.shape[0])])
        return G

    def get_eigenvectors(self, G):
        """
        Calculate the eigenvalues and eigenvectors of the matrix G
        :param G: The array from get_model_matrix
        :return: Eigenvalues and eigenvectors of G
        """
        return np.linalg.eig(G)

    def orthonormalize_model_funcs(self, G, freq):
        """
        Find linear combinations, H,  of the  provided model functions that are orthonormal
        :math:`H_{j}(t) = \frac{1}{\sqrt{\lambda_{j}}} \sum\limits_{k=1}^{k=m}e_{jk}G_{k}(t)`
        :math:`\lambda{j}` is the jth eigenvalue and :math:`e_{jk}` is the kth component of the jth eigenvector of G
        :param G: The array from get_model_matrix
        :param freq: A single frequency (in Hz) from the input frequencies
        :return: Orthonormal model functions evaluated at each time point (shape = (len(model_funcs), len(t))).
        """
        G_t = self.evaluate_model_funcs(freq)
        eigvals, eigvects = self.get_eigenvectors(G)
        H_t = np.array([(1.0 / np.sqrt(eigvals[i])) *
                        np.sum(np.tile(eigvects[:, i].reshape(G_t.shape[0], 1), (1, G_t.shape[1])) * G_t, axis = 0)
                        for i in range(eigvals.shape[0])])
        return H_t

    def get_projections(self, evaluated_model_funcs):
        """
        Project the data on the orthonormal model functions H
        :param evaluated_model_funcs: The array from orthonormalize_model_funcs
        :return: Element-by-element product of the data with the orthonormal model functions
        """
        projected_y = np.tile(self.y.reshape(1, self.y.shape[0]), (evaluated_model_funcs.shape[0], 1)) * \
                              evaluated_model_funcs
        return projected_y

    def get_mean_squared_projections(self, evaluated_model_funcs):
        """
        Sum the projections over time, square them, and take their mean across the model functions
        :param evaluated_model_funcs: From orthonormalize_model_funcs
        :return: Mean squared projection
        """
        projected_y = self.get_projections(evaluated_model_funcs)
        return np.sum(np.sum(projected_y, axis = 1)**2) * (1.0 / projected_y.shape[0])

    def get_posterior(self, evaluated_model_funcs):
        """
        :math:`P \propto [1 - \frac{M\overline{h}^2}{N\overline{d}^2}]^{\frac{M-N}{2}}`
        M = num model functions, N = num observations, :math:`\overline{h}^2` = mean squared projection of observations
        on model functions, :math:`\overline{d}^2` = mean squared observations
        :param evaluated_model_funcs: From orthonormalize_model_funcs
        :return: Marginal posterior probability
        """
        mean_squared_projections = self.get_mean_squared_projections(evaluated_model_funcs)
        num_model_funcs = evaluated_model_funcs.shape[0]
        num_data_points = self.y.shape[0]
        return (1.0 - (num_model_funcs*mean_squared_projections) / (num_data_points*np.mean(self.y**2))) ** (0.5*(num_model_funcs-num_data_points))

    def fit(self, y, t, freqs):
        """
        :param y: Observations (shape = num_time_points) - 1D array/list
        :param t: Time points, measured in seconds (shape = num_time_points) - 1d array/list
        :param freqs: Frequencies for calculation of posterior probabilities, in Hz - 1d array/list
        :return:
        """
        # Get the spacing of the t variables, and calculate the sampling rate
        dt = np.mean(np.ediff1d(t))
        self.sampling_rate = 1.0 / dt
        # Calculate the Nyquist (highest) and fundamental(lowest) detectable frequencies
        self.nyquist = self.sampling_rate / 2.0
        self.fundamental = self.sampling_rate / t.shape[0]

        # Raise warnings if there are frequencies beyond these limits
        if np.max(freqs) > self.nyquist:
            warnings.warn("Maximum frequency for inference greater than Nyquist frequency")
        if np.min(freqs) < self.fundamental:
            warnings.warn("Minimum frequency for inference less than fundamental frequency")
        self.y = np.array(y)
        self.t = np.array(t)
        self.freqs = np.array(freqs)
        posterior = []
        for freq in self.freqs:
            G = self.get_model_matrix(freq)
            H_t = self.orthonormalize_model_funcs(G, freq)
            posterior.append(self.get_posterior(H_t))
        self.posterior = np.nan_to_num(posterior)


class SinusoidMarginal(GeneralMarginal):
    """
    Subclass for marginal posterior inference in a sinusoidal model. 3 methods overridden from GeneralMarginal
    1. get_model_matrix - returns a diagonal identity matrix for this model
    2. orthonormalize_model_funcs - returns the model sinusoids as they are already orthogonal
    3. get_mean_squared_projections - the squared projections are divided by the number of observations, not the
    number of model functions
    For more details, read chapter 2 and 3 of Bretthorst, 1988
    """
    def __init__(self, model_funcs = [sin, cos]):
        super(SinusoidMarginal, self).__init__(model_funcs)

    def get_model_matrix(self, freq):
        return np.identity(2)

    def orthonormalize_model_funcs(self, G, freq):
        G_t = self.evaluate_model_funcs(freq)
        return G_t

    def get_mean_squared_projections(self, evaluated_model_funcs):
        projected_y = self.get_projections(evaluated_model_funcs)
        return np.sum(np.sum(projected_y, axis = 1)**2) * (1.0 / projected_y.shape[1])