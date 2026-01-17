import json
import time
import numpy as np
from numpy.polynomial import polynomial as p
from memory_profiler import memory_usage
from ntt_c import polymul_ntt

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

def gen_poly(xN_1, n, q):
    l = 0
    poly = np.floor(np.random.normal(l, size=n))

    rem = np.polydiv(poly, xN_1)[1] % q

    # Force fixed length
    if len(rem) < n:
        rem = np.pad(rem, (n - len(rem), 0))
    else:
        rem = rem[-n:]

    rem = rem.astype(int)

    while rem[0] == 0:
        rem[0] = int(np.floor(np.random.normal(l)))

    return rem


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

def decryption(xN_1, v, w, q, s, root):
    recovered = polymul_ntt(v,s, q ,root) #p.polymul(v, s) 
    recovered = np.floor(p.polydiv(recovered, xN_1)[1]) % q
    recovered = (w - recovered) % q
    bits = decode_message(recovered, q)
    return bits

deg = [128, 256, 512]  # polynomial degree, so highest is x^3
q_list = [3329, 7681, 12289]
num_runs = 20

def main():
    for i, n in enumerate(deg):
        q = q_list[i]
        print(f"\nRunning degree={n}, q={q}, {num_runs} iterations")

        prime_factors = primes(n)
        primitive_root = find_primitive_root(q,prime_factors)
        
        avg_keygen = 0
        avg_encdec = 0
        avg_mem_peak = 0

        for run in range(num_runs):

            def run_once():
                # modulus
                xN_1 = [1] + [0] * (n - 1) + [1]

                # KEY GENERATION
                start_key_gen = time.perf_counter()
                A = np.floor(np.random.random(size=(n)) * q)
                A = np.floor(p.polydiv(A, xN_1)[1])  # [1] = taking the remainder value
                A = A % q

                # ----ALICE-----
                eA = gen_poly(xN_1, n, q)
                sA = gen_poly(xN_1, n, q)

                # Alice now creates bA = (A x sA) + eA
                bA = polymul_ntt(A, sA, q, primitive_root) #p.polymul(A, sA)
                bA = p.polyadd(bA, eA)
                bA = np.floor(p.polydiv(bA, xN_1)[1]) % q

                # ----BOB-----
                sB = gen_poly(xN_1, n, q)
                eB = gen_poly(xN_1, n, q)

                bB = polymul_ntt(A, sB, q, primitive_root) #p.polymul(A, sB)
                bB = p.polyadd(bB, eB)
                bB = np.floor(p.polydiv(bB, xN_1)[1]) % q

                public_key_Alice = (A, bA)
                public_key_Bob = (A, bB)

                stop_key_gen = time.perf_counter()
                keygen_time = stop_key_gen - start_key_gen

                start_enc_dec = time.perf_counter()
                
                # ENCRYPTION
                # data = {
                #     "msgID": "bsm",
                #     "msgCnt": 42,
                #     "id": "A9B8C7D6",
                #     "secMark": 54321,
                #     "pos": {"lat": 34.0522, "long": -118.2437, "elev": 85.3},
                #     "accuracy": {"semiMajor": 2, "semiMinor": 1},
                #     "motion": {
                #         "speed": 22.5,
                #         "heading": 180.0,
                #         "steeringWheelAngle": 2,
                #     },
                #     "brakes": {
                #         "wheelBrakes": "0000",
                #         "abs": "unavailable",
                #         "traction": "on",
                #     },
                #     "size": {"width": 185, "length": 480},
                # }

                message = json.dumps('hi', separators=(",", ":"))
                m = string_to_bits(message)

                blocks = chunk_bits(m, n)

                r = gen_poly(xN_1, n, q)
                e1 = gen_poly(xN_1, n, q)
                e2 = gen_poly(xN_1, n, q)

                ciphertext = []
                for block in blocks:
                    m_poly = bits_to_poly(block, n)
                    encoded_m = encode_message(m_poly, q)

                    v = polymul_ntt(A, r, q, primitive_root) #p.polymul(A, r)
                    v = p.polyadd(v, e1)
                    v = np.floor(p.polydiv(v, xN_1)[1]) % q

                    w = polymul_ntt(bB, r, q, primitive_root) #p.polymul(bB, r)
                    w = p.polyadd(w, e2)
                    w = p.polyadd(w, encoded_m)
                    w = np.floor(p.polydiv(w, xN_1)[1]) % q

                    ct = (v, w)
                    ciphertext.append(ct)

                # DECRYPTION
                plaintext = []
                for ct in ciphertext:
                    v, w = ct
                    bits = decryption(xN_1, v, w, q, sB, primitive_root)
                    plaintext.extend(bits)

                bits_to_string(plaintext) # print this to see the decrypted message

                stop_enc_dec = time.perf_counter()
                encdec_time = stop_enc_dec - start_enc_dec

                return keygen_time, encdec_time

            mem_usage = memory_usage(run_once, interval=0.01)
            avg_mem_peak += max(mem_usage)
            k_time, e_time = run_once()
            avg_keygen += k_time
            avg_encdec += e_time

        avg_keygen /= num_runs
        avg_encdec /= num_runs
        avg_mem_peak /= num_runs

        print(f"Avg keygen time: {(avg_keygen * 1000):.2f} ms")
        print(f"Avg encrypt/decrypt time: {(avg_encdec * 1000):.2f} ms")
        print(f"Avg peak memory: {avg_mem_peak:.2f} MiB")


if __name__ == "__main__":
    main()