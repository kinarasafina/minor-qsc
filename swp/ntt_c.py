# import numpy as np
# from sympy.discrete.transforms import ntt, intt
# from sympy.discrete.convolutions import convolution_ntt
# from sympy import isprime, nextprime
# import time
# from final_LWE_normal import gen_poly

import numpy as np
import ntt_function
import time

 
# a = np.random.randint(0, q, size=n, dtype=np.int64)
# ntt(a, root, q)

def polymul_ntt(a, b, q, root):
    n = len(a)
    len_c = ((len(a) - 1) + (len(b) - 1)) + 1
    N = 0
    while True:
        N += 1
        if 2**N > len_c:
            break

    n_padded = 2**N

    A = np.zeros(n_padded)
    B = np.zeros(n_padded)
    
    A[:n] = a
    B[:n] = b

    A = np.array(A, dtype=np.int32)
    B = np.array(B, dtype=np.int32)

    ntt_function.ntt(A, root, q)
    ntt_function.ntt(B, root, q)
 
    C = (A * B) % q
 
    inv_root = pow(root, q - 2, q)
    ntt_function.ntt(C, inv_root, q)
 
    inv_n = pow(n_padded, q - 2, q)
    result = (C * inv_n) % q

    return result[:len_c]

q = 577
root = 5     # primitive root for Kyber

# n = 6

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