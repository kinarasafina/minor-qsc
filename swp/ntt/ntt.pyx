# ntt.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
 
cdef long modexp(long a, long e, long mod):
    cdef long r = 1
    while e:
        if e & 1:
            r = (r * a) % mod
        a = (a * a) % mod
        e >>= 1
    return r
 
cpdef ntt(long[:] a, long root, long mod):
    cdef int n = a.shape[0]
    cdef int len_, i, j
    cdef long w, u, v, wlen
 
    len_ = 2
    while len_ <= n:
        wlen = modexp(root, (mod - 1) // len_, mod)
        for i in range(0, n, len_):
            w = 1
            for j in range(len_ // 2):
                u = a[i + j]
                v = (a[i + j + len_ // 2] * w) % mod
                a[i + j] = (u + v) % mod
                a[i + j + len_ // 2] = (u - v + mod) % mod
                w = (w * wlen) % mod
        len_ <<= 1