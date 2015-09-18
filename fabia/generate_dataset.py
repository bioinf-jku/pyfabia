"""Generates a dataset using the Fabia model assumptions.

Author : Thomas Unterthiner
License: GPLv2
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from sklearn.utils import check_random_state


def make_fabia_biclusters(shape, n_clusters, sample_range, feature_range,
                          bg_noise, z_bg_noise, z_mean, z_sd,
                          l_bg_noise, l_mean, l_sd,
                          as_blocks=False, random_state=None):
    ''' Creates a dataset containing biclusters of a given form.
    Parameters
    ----------
    shape: array, shape = [n_samples, n_features]
        shape of the resulting dataset
    n_cluster: integer
        the number of biclusters
    sample_range: array, shape = [2, ]
        min/max number of samples that are part of a bicluster
    feature_range: array, shape = [2, ]
        min/max number of features that are part of a bicluster
    bg_noise: float
        sigma for the background noise
    z_bg_noise: float
        sigma for noise on non-active factors
    z_mean: float
        mean of activated factors
    z_sd: float
        sigma for activated factors
    l_bg_noise: float
        sigma for noise on non-active loadings
    l_mean: float
        mean of activate loadings
    l_sd: float
        sigma for activate loadings
    as_blocks: boolean, optional
        if True, all data belonging to the same bicluster will be in
        one contigeous block. This is most likely only useful for
        illustrative purposes.
      random_state : int seed, RandomState instance, or None (default)
        A pseudo random number generator used by the K-Means
        initialization.
    References
    ----------
    * Hochreiter, Bodenhofer, et. al., 2010. `FABIA: factor analysis
      for bicluster acquisition
      <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2881408/>`__.
    '''
    rng = check_random_state(random_state)
    n, m = shape
    k = n_clusters
    zc = []
    lc = []

    Xn = rng.normal(scale=bg_noise, size=shape)
    X = np.zeros(shape)
    for i in range(k):
        z_noise = rng.normal(scale=z_bg_noise, size=(n, 1))
        z_clean = np.zeros((n, 1))
        z_size = rng.randint(low=sample_range[0], high=sample_range[1] + 1)
        if as_blocks:
            z_start = rng.randint(n - z_size)
            z_idx = range(z_start, z_start + z_size)
        else:
            z_idx = rng.random_integers(0, n-1, z_size)
        z_val = rng.normal(loc=z_mean, scale=z_sd, size=(z_size, 1))
        z_noise[z_idx] = z_val
        z_clean[z_idx] = z_val

        l_noise = rng.normal(scale=l_bg_noise, size=(1, m))
        l_clean = np.zeros((1, m))
        l_size = rng.randint(low=feature_range[0], high=feature_range[1] + 1)
        if as_blocks:
            l_start = rng.randint(m - l_size)
            l_idx = range(l_start, l_start + l_size)
        else:
            l_idx = rng.random_integers(0, m-1, l_size)
        l_val = rng.normal(loc=l_mean, scale=l_sd, size=l_size)
        l_sign = 2 * np.round(rng.random_sample(l_size)) - 1
        l_val = l_val * l_sign
        l_noise[0, l_idx] = l_val
        l_clean[0, l_idx] = l_val

        z_mask = np.zeros((n, ), dtype=np.bool)
        z_mask[z_idx] = True
        zc.append(z_mask)
        l_mask = np.zeros((m, ), dtype=np.bool)
        l_mask[l_idx] = True
        lc.append(l_mask)

        Xn += np.dot(z_noise, l_noise)
        X += np.dot(z_clean, l_clean)

    return (Xn, X, zc, lc)
