# RING-LWE ALGORITHM

# KEY-GEN
import numpy as np
from numpy.polynomial import polynomial as p


n = 4 # polynomial degree, so highest is x^3
# q = 12289
# q = 31
q = 15 # 2^n - 1

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

# ----ALICE-----
print("----ALICE-----")
eA = gen_poly(n, q)
print(eA)
sA = gen_poly(n, q)
print(sA)

# Alice now creates bA = (A x sA) + eA
bA = p.polymul(A,sA)%q
bA = np.floor(p.polydiv(bA,xN_1)[1])
bA = p.polyadd(bA,eA)%q
print(bA)

#----BOB-----
print("----BOB-----")
sB = gen_poly(n,q)
print(sB)
eB = gen_poly(n,q)
print(eB)

bB = p.polymul(A,sB)%q
bB = np.floor(p.polydiv(bB,xN_1)[1])
bB = p.polyadd(bB,eB)%q
print(bB)

#----SHARED----
print("----SHARED----")
sharedAlice = np.floor(p.polymul(sA,bB)%q)
sharedAlice = np.floor(p.polydiv(sharedAlice,xN_1)[1])%q
print(sharedAlice)
# 0, 0, 1, 0

sharedBob = np.floor(p.polymul(sB,bA)%q)
sharedBob = np.floor(p.polydiv(sharedBob,xN_1)[1])%q
print(sharedBob)
# 0, 0, 1, 1