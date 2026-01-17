import numpy as np
from sympy.discrete.transforms import ntt, intt
from sympy.discrete.convolutions import convolution_ntt
from sympy import isprime, nextprime
import time
from final_LWE_normal import gen_poly
from ntt_c import polymul_ntt


# n = 512
# q = 12289
# xN_1 = [1] + [0] * (n - 1) + [1]

# a = gen_poly(xN_1, n, q).astype(int)
# b = gen_poly(xN_1, n, q).astype(int)

# len_c = ((len(a) - 1) + (len(b) - 1)) + 1

# # ------- finding n ----------
# n = 0
# while True:
#     n += 1
#     if 2**n > len_c:
#         break

# #print(n)

# N = 2**n
# # ------- finding prime q ----------
# start = N * len(a) * len(b) + 1
# q = nextprime(start - 1)  # ensure q >= start

# while True:
#     if (q - 1) % N == 0:
#         k = (q - 1) // N
#         break
#     q = nextprime(q)

# print(q)

# start_time = time.perf_counter()
# print('NTT:', convolution_ntt(a,b,q))
# end_time = time.perf_counter()

# print('time NTT:', (end_time - start_time)*1000)

# start_time1 = time.perf_counter()
# print('polymul:', np.polymul(a,b) % q)
# end_time1 = time.perf_counter()

# print('time polymul:' , (end_time1 - start_time1)*1000)

# a = [1, 3, 7, 9, 5, 7, 11, 15, 8, 8]
# b = [9, 1, 2, 3, 4, 2, 3, 7, 6]

def is_primitive_root(u, q, factors):
    phi = q - 1
    for p in factors:
        # check u^(phi/p) mod q != 1
        if pow(u, phi // p, q) == 1:
            return False
    return True

def find_primitive_root(q, factors):
    for u in range(2, q):
        if is_primitive_root(u, q, factors):
            return u
    return None

def primes(n):
    prime_factors = []
    d = 2

    while d * d <= n:
        if n % d == 0:
            prime_factors.append(d)
            while n % d == 0:
                n //= d
        d += 1

    if n > 1 and n not in prime_factors:
        prime_factors.append(n)

    return prime_factors

# Given parameters
deg = 512
Q = 12289

xN_1 = [1] + [0] * (deg - 1) + [1]

a = gen_poly(xN_1, deg, Q).astype(int).tolist()
b = gen_poly(xN_1, deg, Q).astype(int).tolist()

print(a)
print(b)
# print(len(a))
# print(len(b))

prim_root = find_primitive_root(Q, primes(Q - 1))

len_c = ((len(a) - 1) + (len(b) - 1)) + 1

# ------- finding n ----------
n = 0
while True:
    n += 1
    if 2**n > len_c:
        break

# print(n)

N = 2**n
# ------- finding prime q ----------
# start = N * len(a) * len(b) + 1
# q = nextprime(start - 1)  # ensure q >= start

# while True:
#     if (q - 1) % N == 0:
#         k = (q - 1) // N
#         break
#     q = nextprime(q)

# print('q:', q)

# ---- NTT ----
start = time.perf_counter()
c_ntt = convolution_ntt(a, b, Q)
end = time.perf_counter()

print("NTT time (ms):", (end - start) * 1000)

# ---- Naive polynomial multiplication ----
start = time.perf_counter()
c_np = np.polymul(a, b) % Q
end = time.perf_counter()

print("NumPy time (ms):", (end - start) * 1000)

# ------- NTT with c --------
start = time.perf_counter()
c_nttC = polymul_ntt(a,b,Q,prim_root)
end = time.perf_counter()
print("Ntt with C time (ms):", (end - start) * 1000)

# ---- Compare ----
print("Match:", np.all(c_ntt == c_np))
print("Match:", np.all(c_ntt == c_nttC))