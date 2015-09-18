"""Interface to the Fabia Cython Implementation.

Author : Thomas Unterthiner
License: GPL v2

"""
from __future__ import division, absolute_import, print_function

import numpy as np
from sklearn.base import BaseEstimator, BiclusterMixin
from sklearn.utils import check_random_state, as_float_array
from sklearn.utils.validation import assert_all_finite, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.externals.six.moves import xrange
from sklearn.preprocessing import StandardScaler

import scipy.linalg as la
from scipy.sparse import issparse


from ._fabia import fit_fabia


class FabiaBiclustering(BaseEstimator, BiclusterMixin):
    def __init__(self, n_clusters=5, n_iter=500, alpha=0.01,
                 scale=True, spz=0.5, spl=0.0, eps=1e-3, thresZ=0.5,
                 rescale_l=False, random_state=None):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.alpha = alpha
        self.scale = scale
        self.spz = spz
        self.spl = spl
        self.eps = eps
        self.rescale_l = rescale_l
        self.random_state = random_state
        self.thresZ = thresZ
        self._min_lap = 1.0
        self._init_psi = 0.2
        self._spl = 0.0

    def _validate_params(self):
        """Validate input params. """
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be > 0")
        if self.n_iter <= 0:
            raise ValueError("n_iter must be > 0")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        if not (0.5 <= self.spz <= 2.0):
            raise ValueError("spz must be in [0.5, 2]")
        if not (0.0 <= self.spl <= 2.0):
            raise ValueError("spl must be in [0.5, 2]")
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0")
        if not isinstance(self.rescale_l, bool):
            raise ValueError("rescale_l must be either True or False")
        if not isinstance(self.scale, bool):
            raise ValueError("scale must be either True or False")

    def fit(self, X):
        self._validate_params()
        if issparse(X) and self.scale:
            raise ValueError("Cannot scale sparse matrices.")

        X = check_array(X, accept_sparse=['csr', 'csc'])
        if self.scale:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        k = self.n_clusters
        n, m = X.shape
        eps = 10 * np.finfo(X.dtype).eps
        random_state = check_random_state(self.random_state)
        lapla = np.ones(shape=(n, k), dtype=X.dtype)
        Psi = self._init_psi * np.ones(shape=m, dtype=X.dtype)
        L = random_state.normal(size=(k, m)).astype(dtype=X.dtype)
        Z = np.zeros(shape=(n, k), dtype=X.dtype)
        #(L, Z, Psi, lapla) = self._fit(X, L, Z, Psi, lapla)

        n, m = X.shape
        if issparse(X):
            XX = X.copy()
            XX.data **= 2
            XX = (XX.sum(0) / n).A.ravel()
        else:
            #XX = X.var(0)
            XX = (X ** 2).sum(0) / n
        XX[XX < self.eps] = self.eps
        sum2 = np.eye(k, k, dtype=X.dtype)

        (L, Z, Psi, lapla) = fit_fabia(X, L, Z, Psi, lapla, XX, sum2,
                                       self.eps, self.n_iter, self.alpha,
                                       self.spl, self.spz, self._min_lap,
                                       self.rescale_l)

        #scale Z to variance 1
        vz = (Z ** 2).sum(0) / n
        s = np.sqrt(vz + eps)
        Z /= s
        L *= s[:, None]
        lapla *= vz ** 2

        # find appropriate threshold to determine cluster membership
        mom = np.sum((L ** 2).sum(1) * (Z ** 2).sum(0)) / (k * n * m)
        thresL = np.sqrt(mom) / self.thresZ
        self.columns_ = [np.abs(L[i, :]) > thresL for i in range(k)]
        self.rows_ = []
        for i in range(k):
            sp = np.where(Z[:, i] > self.thresZ, Z[:, i], 0).sum()
            sn = -np.where(Z[:, i] < -self.thresZ, Z[:, i], 0).sum()
            if sp > sn:
                self.rows_.append(Z[:, i] > self.thresZ)
            else:
                self.rows_.append(Z[:, i] < -self.thresZ)
        self.Z_ = Z
        self.L_ = L
        self.Psi_ = Psi
        self.lapla_ = lapla
