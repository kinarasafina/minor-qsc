# ntt_function.pyx
cdef long modexp(long a, long e, long mod):
    cdef long r = 1
    while e:
        if e & 1:
            r = (r * a) % mod
        a = (a * a) % mod
        e >>= 1
    return r

cdef int reverse_bits(int num, int log_n):
    """Reverse the bits of num using log_n bits"""
    cdef int result = 0
    cdef int i
    for i in range(log_n):
        result = (result << 1) | (num & 1)
        num >>= 1
    return result

cpdef ntt(long[:] a, long root, long mod):
    cdef int n = a.shape[0]
    cdef int len_, i, j, log_n, rev
    cdef long w, u, v, wlen, temp
    
    # Bit-reversal permutation
    log_n = 0
    temp = n
    while temp > 1:
        log_n += 1
        temp >>= 1
    
    for i in range(n):
        rev = reverse_bits(i, log_n)
        if i < rev:
            temp = a[i]
            a[i] = a[rev]
            a[rev] = temp
    
    # Cooley-Tukey NTT
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