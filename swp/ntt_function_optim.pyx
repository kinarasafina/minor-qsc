# ntt_function_optim.pyx
import numpy as np

cdef long modexp(long a, long e, long mod):
    """Fast modular exponentiation"""
    cdef long r = 1
    a = a % mod
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

cdef void ntt_core(long[:] a, long root, long mod):
    """Core NTT implementation with bit-reversal"""
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

cpdef long[:] polymul_ntt_ring(long[:] a, long[:] b, long q, long psi):
    """
    Polynomial multiplication in Z_q[x]/(x^n + 1) using NTT.
    
    Parameters:
    -----------
    a, b : input polynomials (must be same length and power of 2)
    q : modulus
    psi : 2n-th primitive root of unity mod q
    
    Returns:
    --------
    result : a * b mod (x^n + 1) in Z_q
    """
    cdef int n = a.shape[0]
    cdef int i
    cdef long psi_power, inv_psi_power, inv_n
    
    # Allocate working arrays
    cdef long[:] a_transformed = np.empty(n, dtype=np.int32)
    cdef long[:] b_transformed = np.empty(n, dtype=np.int32)
    cdef long[:] result = np.empty(n, dtype=np.int32)
    
    # Step 1: Forward transformation with psi powers
    # Multiply a[i] by psi^i and b[i] by psi^i
    psi_power = 1
    for i in range(n):
        a_transformed[i] = (a[i] * psi_power) % q
        b_transformed[i] = (b[i] * psi_power) % q
        psi_power = (psi_power * psi) % q
    
    # Step 2: Apply NTT using psi^2 as the primitive n-th root
    cdef long root = (psi * psi) % q
    ntt_core(a_transformed, root, q)
    ntt_core(b_transformed, root, q)
    
    # Step 3: Pointwise multiplication
    for i in range(n):
        result[i] = (a_transformed[i] * b_transformed[i]) % q
    
    # Step 4: Inverse NTT
    cdef long inv_root = modexp(root, q - 2, q)
    ntt_core(result, inv_root, q)
    
    # Step 5: Inverse transformation with psi^(-i) and normalize by n
    inv_n = modexp(n, q - 2, q)
    inv_psi_power = 1
    cdef long inv_psi = modexp(psi, q - 2, q)
    
    for i in range(n):
        result[i] = (result[i] * inv_psi_power % q) * inv_n % q
        inv_psi_power = (inv_psi_power * inv_psi) % q
    
    return result

cpdef long find_primitive_root_2n(long q, int n):
    """
    Find a 2n-th primitive root of unity modulo q.
    For this to exist, q must be of the form q = k*2n + 1.
    
    Parameters:
    -----------
    q : prime modulus
    n : polynomial degree (must be power of 2)
    
    Returns:
    --------
    psi : 2n-th primitive root of unity mod q
    """
    cdef long phi = q - 1  # Euler's totient for prime q
    cdef long exponent = phi // (2 * n)
    cdef long candidate, psi
    cdef int i
    
    # Try small candidates
    for candidate in range(2, min(100, q)):
        psi = modexp(candidate, exponent, q)
        
        # Check if psi^(2n) = 1 and psi^n = -1 (equivalently, q-1)
        if modexp(psi, 2 * n, q) == 1 and modexp(psi, n, q) == q - 1:
            return psi
    
    return 0  # Not found