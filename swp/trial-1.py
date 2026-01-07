# RING-LWE ALGORITHM

# KEY-GEN
import numpy as np
from numpy.polynomial import polynomial as p


n = 10 # polynomial degree, so highest is x^3
# q = 12289
# q = 31
q = 1023 # 2^n - 1

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


A = np.floor(np.random.random(size=(n)) * q)
A = np.floor(p.polydiv(A, xN_1)[1])  # [1] = taking the remainder value
A = A % q
print(A)

# ----ALICE-----
print("----ALICE-----")
eA = gen_poly(n, q)
print(eA)
sA = gen_poly(n, q)
print(sA)

# Alice now creates bA = (A x sA) + eA
bA = p.polymul(A,sA)
bA = p.polyadd(bA,eA)
bA = np.floor(p.polydiv(bA,xN_1)[1])%q
print(bA)

#----BOB-----
print("----BOB-----")
sB = gen_poly(n,q)
print(sB)
eB = gen_poly(n,q)
print(eB)

bB = p.polymul(A,sB)
bB = p.polyadd(bB,eB)
bB = np.floor(p.polydiv(bB,xN_1)[1])%q
print(bB)

#----SHARED----
print("----SHARED----")
sharedAlice = np.floor(p.polymul(sA,bB))
sharedAlice = np.floor(p.polydiv(sharedAlice,xN_1)[1])%q
print(sharedAlice)

sharedBob = np.floor(p.polymul(sB,bA))
sharedBob = np.floor(p.polydiv(sharedBob,xN_1)[1])%q
print(sharedBob)


#----VERIFICATION----
def construct_u_helper(poly, q):
    bits = []
    q4 = q / 4
    q2 = q / 2
    q34 = 3 * q / 4

    for c in poly:
        if c < q4:
            bits.append(0)
        elif c < q2:
            bits.append(1)
        elif c < q34:
            bits.append(0)
        else:
            bits.append(1)

    return bits

u = construct_u_helper(sharedBob, q)
print(u)

# --Bob
i = 0
while (i < len(u)):
	#Region 0 (0 --- q/4 and q/2 --- 3q/4)
	if (u[i] == 0):
		if (sharedBob[i] >= q*0.125 and sharedBob[i] < q*0.625):
			sharedBob[i] = 1
		else:
			sharedBob[i] = 0


	#Region 1 (q/4 --- q/2 and 3q/4 --- q)
	elif (u[i] == 1):
		if (sharedBob[i] >= q*0.875 and sharedBob[i] < q*0.375):
			sharedBob[i] = 0
		else:
			sharedBob[i] = 1

	else:
		print("error! (2)")

	i += 1

#--Alice
i = 0
while (i < len(u)):
	#Region 0 (0 --- q/4 and q/2 --- 3q/4)
	if (u[i] == 0):
		if (sharedAlice[i] >= q*0.125 and sharedAlice[i] < q*0.625):
			sharedAlice[i] = 1
		else:
			sharedAlice[i] = 0


	#Region 1 (q/4 --- q/2 and 3q/4 --- q)
	elif (u[i] == 1):
		if (sharedAlice[i] >= q*0.875 and sharedAlice[i] < q*0.375):
			sharedAlice[i] = 0
		else:
			sharedAlice[i] = 1

	else:
		print("error! (3)")
	i += 1
	
print(sharedAlice)
print(sharedBob)
