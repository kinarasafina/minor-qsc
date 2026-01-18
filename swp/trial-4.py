import json
import time
import numpy as np
from numpy.polynomial import polynomial as p
from ntt_c import polymul_ntt, polymul_ntt_optim

def gen_poly(n, q):
    """Generate a polynomial with coefficients from normal distribution"""
    poly = np.floor(np.random.normal(0, size=n)).astype(np.int64)
    poly = poly % q
    
    # Ensure first coefficient is non-zero
    while poly[0] == 0:
        poly[0] = int(np.floor(np.random.normal(0))) % q
    
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
    poly = np.zeros(n, dtype=np.int64)
    for i in range(len(bits)):
        poly[i] = bits[i]
    return poly

def encode_message(m, q):
    return np.array([(q // 2) * int(mi) for mi in m], dtype=np.int64)

def decode_message(poly, q):
    bits = []
    for c in poly:
        if q / 4 < c < 3 * q / 4:
            bits.append(1)
        else:
            bits.append(0)
    return bits

def decryption(v, w, q, s, n, root):
    """Decrypt using NTT-based polynomial multiplication"""
    # NO POLYDIV NEEDED! The NTT function handles x^n + 1 reduction
    recovered = polymul_ntt_optim(v, s, q, root, n)
    recovered = (w - recovered) % q
    bits = decode_message(recovered, q)
    return bits

deg = [128, 256, 512]
q_list = [3329, 7681, 12289]
PRIMITIVE_ROOTS = {3329: 17, 7681: 62, 12289: 11}
num_runs = 20

def main():
    for i, n in enumerate(deg):
        q = q_list[i]
        print(f"\nRunning degree={n}, q={q}, {num_runs} iterations")
        
        root = PRIMITIVE_ROOTS[q]
        avg_keygen = 0
        avg_encdec = 0
        
        for run in range(num_runs):
            
            def run_once():
                # KEY GENERATION
                start_key_gen = time.perf_counter()
                
                # Generate random polynomial A
                A = np.random.randint(0, q, size=n, dtype=np.int64)
                
                # ----ALICE-----
                eA = gen_poly(n, q)
                sA = gen_poly(n, q)
                
                # Alice creates bA = (A * sA) + eA mod (x^n + 1)
                # NO POLYDIV NEEDED!
                bA = polymul_ntt_optim(A, sA, q, root, n)
                bA = (bA + eA) % q
                
                # ----BOB-----
                sB = gen_poly(n, q)
                eB = gen_poly(n, q)
                
                # Bob creates bB = (A * sB) + eB mod (x^n + 1)
                # NO POLYDIV NEEDED!
                bB = polymul_ntt_optim(A, sB, q, root, n)
                bB = (bB + eB) % q
                
                stop_key_gen = time.perf_counter()
                keygen_time = stop_key_gen - start_key_gen
                
                start_enc_dec = time.perf_counter()
                
                # ENCRYPTION
                data = {
                    "msgID": "bsm",
                    "msgCnt": 42,
                    "id": "A9B8C7D6",
                    "secMark": 54321,
                    "pos": {"lat": 34.0522, "long": -118.2437, "elev": 85.3},
                    "accuracy": {"semiMajor": 2, "semiMinor": 1},
                    "motion": {
                        "speed": 22.5,
                        "heading": 180.0,
                        "steeringWheelAngle": 2,
                    },
                    "brakes": {
                        "wheelBrakes": "0000",
                        "abs": "unavailable",
                        "traction": "on",
                    },
                    "size": {"width": 185, "length": 480},
                }
                
                message = json.dumps(data, separators=(",", ":"))
                m = string_to_bits(message)
                blocks = chunk_bits(m, n)
                
                r = gen_poly(n, q)
                e1 = gen_poly(n, q)
                e2 = gen_poly(n, q)
                
                ciphertext = []
                for block in blocks:
                    m_poly = bits_to_poly(block, n)
                    encoded_m = encode_message(m_poly, q)
                    
                    # v = A * r + e1 mod (x^n + 1)
                    # NO POLYDIV NEEDED!
                    v = polymul_ntt_optim(A, r, q, root, n)
                    v = (v + e1) % q
                    
                    # w = bB * r + e2 + encoded_m mod (x^n + 1)
                    # NO POLYDIV NEEDED!
                    w = polymul_ntt_optim(bB, r, q, root, n)
                    w = (w + e2 + encoded_m) % q
                    
                    ct = (v, w)
                    ciphertext.append(ct)
                
                # DECRYPTION
                plaintext = []
                for ct in ciphertext:
                    v, w = ct
                    bits = decryption(v, w, q, sB, n, root)
                    plaintext.extend(bits)
                
                final_text = bits_to_string(plaintext)
                
                stop_enc_dec = time.perf_counter()
                encdec_time = stop_enc_dec - start_enc_dec
                
                return keygen_time, encdec_time, final_text
            
            k_time, e_time, final_text = run_once()
            avg_keygen += k_time
            avg_encdec += e_time
        
        avg_keygen /= num_runs
        avg_encdec /= num_runs
        
        print(f'Decrypted text: {final_text}')
        print(f"Avg keygen time: {(avg_keygen * 1000):.2f} ms")
        print(f"Avg encrypt/decrypt time: {(avg_encdec * 1000):.2f} ms")

if __name__ == "__main__":
    main()