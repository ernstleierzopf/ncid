import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword


class Portax(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        return generate_random_keyword(self.alphabet, length)

    def encrypt(self, plaintext, key):
        return enc_dec(plaintext, key)

    def decrypt(self, ciphertext, key):
        return enc_dec(ciphertext, key)


leng = 26
a2_keys = np.array([[i % leng for i in range(0, leng * 2, 2)], [(i + 1) % leng for i in range(0, leng * 2, 2)]])


def enc_dec(text, key):
    text = list(text)
    while len(text) % 2 != 0:
        text.append(23)  # b'x' = pos 23 in alphabet
    text2 = [0] * len(text)
    for i in range(int(len(text) / 2)):
        pos = int(i / len(key)) * len(key) + i
        t0 = text[pos]
        if pos + len(key) < len(text):
            t1 = text[pos + len(key)]
        else:
            t1 = text[pos + int((len(text) % len(key)) / 2)]
        k = int(key[pos % len(key)] / 2)
        pos1 = np.where(a2_keys[t1 % 2] == t1)[0]
        if pos1[0] <= k + t0:
            pos1 = pos1[int(t0 / 13)]
        else:
            pos1 = pos1[0]
        pos2 = a2_keys[t1 % 2]
        if t0 < 13:
            pos1 -= k
            pos2 = pos2[(k + t0) % leng]
        else:
            pos2 = pos2[t0]
        if pos1 < 0:
            pos1 = pos1 % 13
        if pos1 == t0 and pos2 == t1:
            if pos1 > pos2:
                pos1 = t0 - 13 - k + (t1 - 1 <= 0 and pos1 < 13) * (13 + 2 * k)
                pos2 = t1 - 1 + 2 * (t1 - 1 < 0)
            else:
                pos1 = (t0 + 13 + k) % leng
                pos2 = t1 + 1 - 2 * (t1 + 1 >= leng)
        if pos1 < 0:
            pos1 = pos1 % 13
        text2[pos] = pos1
        if pos + len(key) < len(text):
            text2[pos + len(key)] = pos2
        else:
            text2[pos + int((len(text) % len(key)) / 2)] = pos2
    return np.array(text2)
