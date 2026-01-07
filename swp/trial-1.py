# RING-LWE ALGORITHM

# KEY-GEN
import numpy as np
from numpy.polynomial import polynomial as p


n = 4
# q = 12289
q = 31

# modulus
xN_1 = [1] + [0] * (n - 1) + [1]
print(xN_1)  # x^4 + 0x^3 + 0x^2 + 0x + 1


def gen_poly(n, q):
    global xN_1
    l = 0  # Gamma Distribution Location (Mean "center" of dist.)
    poly = np.floor(np.random.normal(l, size=(n)))
    poly = np.floor(p.polydiv(poly, xN_1)[1] % q)

    if len(poly) < n:
        poly = np.pad(poly, (n - len(poly), 0))
    else:
        poly = poly[:n]

    return poly


A = np.floor(np.random.random(size=(n)) * q) % q
A = np.floor(p.polydiv(A, xN_1)[1])  # [1] = taking the remainder value
print(A)

eA = gen_poly(n, q)
print(eA)
sA = gen_poly(n, q)
print(sA)
