# KEY-GEN
import time
import numpy as np
from numpy.polynomial import polynomial as p


deg = [128, 256, 512]  # polynomial degree, so highest is x^3
# q = 12289
# q = 31

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

def string_to_bits(s):
        return [int(bit) for byte in s.encode("utf-8") for bit in format(byte, "08b")]

def bits_to_string(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i : i + 8]
        if len(byte) < 8:
            break
        chars.append(chr(int("".join(str(b) for b in byte), 2)))
    return "".join(chars)

def chunk_bits(bits, n):
        return [bits[i : i + n] for i in range(0, len(bits), n)]

def bits_to_poly(bits, n):
        poly = np.zeros(n, dtype=int)
        for i in range(len(bits)):
            poly[i] = bits[i]
        return poly

def encode_message(m, q):
    return np.array([(q // 2) * int(mi) for mi in m], dtype=object)

def decode_message(poly, q):
        bits = []
        for c in poly:
            if q / 4 < c < 3 * q / 4:
                bits.append(1)
            else:
                bits.append(0)
        return bits

def decryption(v, w, q, s):
    recovered = p.polymul(v, s)
    recovered = np.floor(p.polydiv(recovered, xN_1)[1]) % q
    recovered = (w - recovered) % q
    bits = decode_message(recovered, q)
    return bits

for n in deg:
    q = 2**n - 1
    # modulus
    xN_1 = [1] + [0] * (n - 1) + [1]
    # print(xN_1)  # x^4 + 0x^3 + 0x^2 + 0x + 1

    # KEY GENERATION
    start_key_gen = time.perf_counter()
    A = np.floor(np.random.random(size=(n)) * q)
    A = np.floor(p.polydiv(A, xN_1)[1])  # [1] = taking the remainder value
    A = A % q
    # print(A)

    # ----ALICE-----
    # print("----ALICE-----")
    eA = gen_poly(n, q)
    # print(eA)
    sA = gen_poly(n, q)
    # print(sA)

    # Alice now creates bA = (A x sA) + eA
    bA = p.polymul(A, sA)
    bA = p.polyadd(bA, eA)
    bA = np.floor(p.polydiv(bA, xN_1)[1]) % q
    # print(bA)

    # ----BOB-----
    # print("----BOB-----")
    sB = gen_poly(n, q)
    # print(sB)
    eB = gen_poly(n, q)
    # print(eB)

    bB = p.polymul(A, sB)
    bB = p.polyadd(bB, eB)
    bB = np.floor(p.polydiv(bB, xN_1)[1]) % q
    # print(bB)

    public_key_Alice = (A, bA)
    public_key_Bob = (A, bB)
    stop_key_gen = time.perf_counter()

    print(f"Key generation time: {(stop_key_gen - start_key_gen) * 1000}")

    start_enc_dec = time.perf_counter()
    # ENCRYPTION
    k = 200

    message = "hi"
    m = string_to_bits(message)
    # print(m)

    blocks = chunk_bits(m, n)
    # print(blocks)

    r = gen_poly(n, q)
    e1 = gen_poly(n, q)
    e2 = gen_poly(n, q)

    ciphertext = []
    for block in blocks:
        m_poly = bits_to_poly(block, n)
        encoded_m = encode_message(m_poly, q)
        # print('m_poly:',m_poly)
        # print(encoded_m)

        v = p.polymul(A, r)
        v = p.polyadd(v, e1)
        v = np.floor(p.polydiv(v, xN_1)[1]) % q

        w = p.polymul(bB, r)
        w = p.polyadd(w, e2)
        w = p.polyadd(w, encoded_m)
        w = np.floor(p.polydiv(w, xN_1)[1]) % q

        ct = (v, w)
        ciphertext.append(ct)
    # print(ciphertext)

    # DECRYPTION
    plaintext = []
    for ct in ciphertext:
        v, w = ct
        bits = decryption(v, w, q, sB)
        plaintext.extend(bits)

    # print(plaintext)
    bits_to_string(plaintext)

    stop_enc_dec = time.perf_counter()

    print(f"Encrypt Decrypt Time: {(stop_enc_dec - start_enc_dec) * 1000}")
