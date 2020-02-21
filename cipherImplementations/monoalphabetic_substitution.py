import random
import numpy as np

alphabet = b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
unknown_symbol = b'?'
unknown_symbol_number = 90

def generate_random_key(length=None):
    global alphabet
    alphabet2 = b'' + alphabet
    key = b''
    for i in range(len(alphabet)):
        position = int(random.randrange(0, len(alphabet2)))
        char = bytes([alphabet2[position]])
        key = key + char
        alphabet2 = alphabet2.replace(char, b'')
    return key

def encrypt(plaintext, key):
    global alphabet
    ciphertext = []
    for position in range(0, len(plaintext)):
        p = plaintext[position]
        if (p > len(alphabet)):
            ciphertext.append(unknown_symbol_number)
            continue
        c = np.where(key == p)[0][0]
        ciphertext.append(c)
    return np.array(ciphertext)