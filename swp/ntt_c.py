# import numpy as np
# from sympy.discrete.transforms import ntt, intt
# from sympy.discrete.convolutions import convolution_ntt
# from sympy import isprime, nextprime
# import time
# from final_LWE_normal import gen_poly

import numpy as np
import ntt_function
import ntt_function_optim
import time

PSI_VALUES = {
    (3329, 128): 17,      # For q=3329, n=128
    (3329, 256): 49,      # For q=3329, n=256
    (7681, 256): 62,      # For q=7681, n=256
    (12289, 256): 49,     # For q=12289, n=256
    (12289, 512): 11,     # For q=12289, n=512
}

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

def polymul_ntt_optim(a, b, q, n=None):
    n = len(a)
    len_c = ((len(a) - 1) + (len(b) - 1)) + 1
    N = 0
    while True:
        N += 1
        if 2**N > len_c:
            break

    n_padded = 2**N
    # #---------------------
    # if n is None:
    #     n = len(a)
    
    # # Ensure inputs are the right size (must be power of 2)
    # if not (n & (n - 1) == 0):  # Check if n is power of 2
    #     # Pad to next power of 2
    #     n_padded = 1
    #     while n_padded < n:
    #         n_padded <<= 1
    #     n = n_padded
    
    # Convert to numpy arrays and pad if necessary
    A = np.zeros(n_padded, dtype=np.int32)
    B = np.zeros(n_padded, dtype=np.int32)
    
    A[:n] = a
    B[:n] = b

    # Get or compute 2n-th primitive root
    if (q, n) in PSI_VALUES:
        psi = PSI_VALUES[(q, n)]
    else:
        # Compute it (slower, but works for any valid q, n)
        psi = ntt_function_optim.find_primitive_root_2n(q, n_padded)
        if psi == 0:
            raise ValueError(f"Could not find 2n-th primitive root for q={q}, n={n}")
        # print(f"Computed psi={psi} for q={q}, n={n}")
    
    # Perform NTT-based multiplication
    result = np.asarray(ntt_function_optim.polymul_ntt_ring(A, B, q, psi))
    
    return result

q = 577
root = 5     # primitive root for Kyber

n = 6

a = [1, 3, 7, 9, 5, 7]
b = [9, 1, 2, 3, 4, 2]

start_time = time.perf_counter()
c_ntt = polymul_ntt(a,b,q,root)
end_time = time.perf_counter()

print(f'{(end_time - start_time)*1000} ms')

start_time = time.perf_counter()
c_pm = np.polymul(a, b) % q
end_time = time.perf_counter()

print(f'{(end_time - start_time)*1000} ms')   

start_time = time.perf_counter()
c_pm_opt = polymul_ntt_optim(a,b,q)
end_time = time.perf_counter()

print(f'{(end_time - start_time)*1000} ms')   

print(c_pm, c_pm_opt)