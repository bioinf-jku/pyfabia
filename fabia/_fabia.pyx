"""Implements the FABIA biclustering algorithm in Cython.

Author : Thomas Unterthiner
License: GPLv2
"""

cimport numpy as np
cimport cython
from scipy.linalg.cython_blas cimport sgemm, dgemm, sgemv, dgemv
from scipy.linalg.cython_lapack cimport spotri, dpotri, spotrf, dpotrf


from libc.math cimport fabs, sqrt, copysign
import numpy as np

ctypedef fused real_t:
    np.float32_t
    np.float64_t

np.import_array()
import scipy.linalg as la


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void add_dot(real_t[:, ::1] a, real_t[:, ::1] b, real_t[:, ::1] out,
            int transA=False, int transB=False,
            real_t alpha=1.0, real_t beta=1.0) nogil:
    cdef int m, k, l, n, lda, ldb, ldc
    cdef char* ta
    cdef char* tb

    if transB:
        m, k, tb = b.shape[0], b.shape[1], "t"
    else:
        m, k, tb = b.shape[1], b.shape[0], "n"
    if transA:
        l, n, ta = a.shape[0], a.shape[1], "t"
    else:
        l, n, ta = a.shape[1], a.shape[0], "n"
    lda = a.shape[1]
    ldb = b.shape[1]
    ldc = out.shape[1]

    if real_t is np.float32_t:
        sgemm(tb, ta, &m, &n, &k, &alpha,
           &(b[0, 0]), &ldb, &(a[0, 0]), &lda, &beta,
           &(out[0, 0]), &ldc)
    else:
        dgemm(tb, ta, &m, &n, &k, &alpha,
           &(b[0, 0]), &ldb, &(a[0, 0]), &lda, &beta,
           &(out[0, 0]), &ldc)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void add_dot_vec(real_t[:, ::1] a, real_t[::1] x, real_t[::1] out,
                      int transA, real_t alpha, real_t beta) nogil:

    cdef int m, n
    cdef char* ta = "n" if transA else "t"  # invert f-order
    cdef int lda = a.shape[1]
    cdef int inc = 1
    n = a.shape[0]
    m = a.shape[1]
    if real_t is np.float32_t:
        sgemv(ta, &m, &n, &alpha, &(a[0, 0]), &lda,
              &(x[0]), &inc, &beta, &(out[0]), &inc)
    else:
        dgemv(ta, &m, &n, &alpha, &(a[0, 0]), &lda,
              &(x[0]), &inc, &beta, &(out[0]), &inc)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void invert_pos(real_t[:, ::1] a) nogil:
    cdef int info, i, j
    cdef int n = a.shape[0]
    cdef int lda = a.shape[1]
    cdef char* uplo = "l"
    if real_t is np.float32_t:
        spotrf(uplo, &n, &(a[0, 0]), &lda, &info)
        spotri(uplo, &n, &(a[0, 0]), &lda, &info)
    else:
        dpotrf(uplo, &n, &(a[0, 0]), &lda, &info)
        dpotri(uplo, &n, &(a[0, 0]), &lda, &info)
    for i in range(a.shape[0]):
        for j in range(i+1, a.shape[1]):
            a[j, i] = a[i, j]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _estimate_z(real_t[:, ::1] X,
                      real_t[:, ::1] Z,
                      int j,
                      real_t[:, ::1] lpsi,
                      real_t[:, ::1] lpsil,
                      real_t[:, ::1] lapla,
                      real_t[:, ::1] sum2,
                      int only_z,
                      real_t min_lap,
                      real_t spz,
                      real_t[:, ::1] lpsil_inv,
                      real_t[:, ::1] lpsil_inv_lpsi) nogil:
    cdef int i, k
    lpsil_inv[:] = lpsil
    for i in range(lpsil_inv.shape[0]):
        lpsil_inv[i, i] += lapla[j, i]
    invert_pos(lpsil_inv)
    #inv = la.inv(lpsil + np.diag(lapla[j, ]))
    #np.dot(lpsil_inv, lpsi, out=lpsil_inv_lpsi)
    add_dot(lpsil_inv, lpsi, lpsil_inv_lpsi, False, False, 1.0, 0.0)
    #print(inv.is_c_contig(), inv.is_f_contig(), w.flags, lapla.flags)
    #print(Z.shape[1], w.shape[0], w.shape[1], X.shape[1])
    #Z[j, :] = np.dot(lpsil_inv_lpsi, X[j, :])
    add_dot_vec(lpsil_inv_lpsi, X[j], Z[j], False, 1.0, 0.0)
    if only_z:
        return
    #lpsil_inv += np.outer(Z[j, ], Z[j, ])
    for i in range(lpsil_inv.shape[0]):
        for k in range(lpsil_inv.shape[1]):
            lpsil_inv[i, k] += Z[j, i] * Z[j, k]
    for i in range(sum2.shape[0]):
        for k in range(sum2.shape[1]):
            sum2[i, k] += lpsil_inv[i, k]

    #eps = 10 * np.finfo(X.dtype).eps
    for i in range(lapla.shape[1]):
        lapla[j, i] = (1e-10 + lpsil_inv[i, i]) ** (-spz)
        if lapla[j, i] < min_lap:
            lapla[j, i] = min_lap

cdef inline real_t sign(real_t x) nogil:
    return (0.0 < x) - (x < 0.0)


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_fabia(real_t[:, ::1] X, real_t[:, ::1] L, real_t[:, ::1] Z, real_t[::1] Psi,
         real_t[:, ::1] lapla, real_t[::1] XX, real_t[:, ::1] sum2, real_t eps, int n_iter,
         real_t alpha, real_t spl, real_t spz, real_t min_lap, int rescale_l):
        cdef int k, n, m, cyc, j, i
        cdef real_t dL, s
        cdef real_t[:, ::1] lpsi, lpsil, lpsil_inv, lpsil_inv_lpsi, inv_psi, xz
        n = X.shape[0]
        m = X.shape[1]
        k = L.shape[0]
        if real_t is np.float32_t:
            dtype = np.float32
        else:
            dtype = np.float64

        lpsi = np.zeros((k, m), dtype=dtype)
        lpsil = np.zeros((k, k), dtype=dtype)
        inv_psi = np.zeros((m, m), dtype=dtype)
        lpsil_inv = np.zeros((k, k), dtype=dtype)
        lpsil_inv_lpsi = np.zeros((k, m), dtype=dtype)
        xz = np.zeros((m, k), dtype=dtype)

        #eps = 10 * np.finfo(X.dtype).eps
        with nogil:
            for cyc in range(n_iter):
                for i in range(sum2.shape[0]):
                    for j in range(sum2.shape[1]):
                        sum2[i, j] = 0.0
                    sum2[i, i] = 1e-6
                #sum2 = eps * np.eye(k, k, dtype=X.dtype)
                #lpsi = np.dot(L, np.diag(1.0 / Psi))
                for i in range(inv_psi.shape[0]):
                    for j in range(inv_psi.shape[1]):
                        inv_psi[i, j] = 0.0
                    inv_psi[i, i] = 1.0 / Psi[i]
                #lpsi = np.dot(L, inv_psi)
                #lpsil = np.dot(lpsi, L.T)
                add_dot(L, inv_psi, lpsi, False, False, 1.0, 0.0)
                add_dot(lpsi, L, lpsil, False, True, 1.0, 0.0)
                for j in xrange(X.shape[0]):
                    _estimate_z(X, Z, j, lpsi, lpsil, lapla, sum2, 0, min_lap, spz, lpsil_inv, lpsil_inv_lpsi)
                #xz = np.dot(X.T, Z)
                add_dot(X, Z, xz, True, False, 1.0, 0.0)
                #sll = la.inv(sum2)
                invert_pos(sum2)
                add_dot(sum2, xz, L, False, True, 1.0, 0.0)
                #L = np.dot(sum2, xz.T)
                #dL = self.alpha * Psi * L
                for i in range(L.shape[0]):
                    for j in range(L.shape[1]):
                        dL = alpha * Psi[j]* sign(L[i, j]) * fabs(L[i, j]) ** (-spl)
                        #dL = alpha * Psi * np.sign(L) * np.abs(L) ** (-spl)
                        #L = np.where(abs(L) > abs(dL), L - dL, 0)
                        if fabs(L[i, j]) > fabs(dL):
                            L[i, j] -= dL
                        else:
                            L[i, j] = 0
                #Psi = np.abs(XX - (L * xz.T).sum(0) / n)
                for i in range(m):
                    s = 0
                    for j in range(k):
                        s += L[j, i] * xz[i, j]
                    Psi[i] = fabs(XX[i] - s/n)
                    if Psi[i] < eps:
                        Psi[i] = eps
                #scale L to variance 1
                if rescale_l:
                    for i in range(L.shape[0]):
                        s = 0
                        for j in range(L.shape[1]):
                            s += L[i, j]
                        s = sqrt(s / L.shape[1] + 1e-6)
                        for j in range(L.shape[1]):
                            L[i, j] /= s
                        for j in range(lapla.shape[1]):
                            lapla[i, j] *= (s ** 2) ** -spz
                    #s = (L * L).sum(1)
                    #s = np.sqrt(s / m + 1e-10)
                    #L /= s[:, None]
                    #s = (s ** 2) ** -self.spz
                    #lapla *= s

            # last Z update
            for j in range(X.shape[0]):
                _estimate_z(X, Z, j, lpsi, lpsil, lapla, sum2, 0, min_lap, spz, lpsil_inv, lpsil_inv_lpsi)
        return (np.asarray(L), np.asarray(Z), np.asarray(Psi), np.asarray(lapla))
