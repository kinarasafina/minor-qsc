import numpy as np
import ntt_function

def polymul_ntt(a, b, q, root):
    n = len(a)
    len_c = ((len(a) - 1) + (len(b) - 1)) + 1
    N = 0
    while True:
        N += 1
        if 2**N > len_c:
            break

    n_padded = 2**N

    A = np.zeros(n_padded, dtype=np.int32)
    B = np.zeros(n_padded, dtype=np.int32)
    
    A[:n] = a
    B[:n] = b

    # A = np.array(A, dtype=np.int32)
    # B = np.array(B, dtype=np.int32)

    ntt_function.ntt(A, root, q)
    ntt_function.ntt(B, root, q)
 
    C = (A * B) % q
 
    inv_root = pow(root, q - 2, q)
    ntt_function.ntt(C, inv_root, q)
 
    inv_n = pow(n_padded, q - 2, q)
    result = (C * inv_n) % q

    return result[:len_c]