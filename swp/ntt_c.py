# import numpy as np
# from sympy.discrete.transforms import ntt, intt
# from sympy.discrete.convolutions import convolution_ntt
# from sympy import isprime, nextprime
# import time
# from final_LWE_normal import gen_poly

import numpy as np
import ntt as z
 
q = 577
root = 5     # primitive root for Kyber
n = 6
 
# a = np.random.randint(0, q, size=n, dtype=np.int64)
# ntt(a, root, q)

def polymul_ntt(a, b, q, root):
    n = len(a)
    A = a.copy()
    B = b.copy()
 
    z.ntt(A, root, q)
    z.ntt(B, root, q)
 
    C = (A * B) % q
 
    inv_root = pow(root, q - 2, q)
    z.ntt(C, inv_root, q)
 
    inv_n = pow(n, q - 2, q)
    return (C * inv_n) % q

a = [1, 3, 7, 9, 5, 7]
b = [9, 1, 2, 3, 4, 2]

import ntt
print(ntt.__file__)  # should point to ntt.cp311-win_amd64.pyd
print(polymul_ntt(a,b,q,root))
