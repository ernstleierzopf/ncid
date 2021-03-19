import numpy as np
from cipherImplementations.playfair import Playfair
from cipherImplementations.cipher import generate_random_keyword, generate_keyword_alphabet
from util.utils import remove_unknown_symbols


class SeriatedPlayfair(Playfair):
    def generate_random_key(self, length):
        return [generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length)), length]

    def encrypt(self, plaintext, key):
        ciphertext = []
        pt = []
        shift1 = 0
        shift2 = 0
        for i in range(len(plaintext) - key[1]*3 + 1):
            if i % key[1] == 0 and shift1 != shift2:
                shift1 = shift2
                shift2 = 0
            pos = int(i / key[1]) * key[1] * 2 + i % key[1]
            pos1 = pos - shift1
            pos2 = pos + key[1] - shift2 - shift1
            if pos2 >= len(plaintext):
                break
            pt.append(plaintext[pos1])
            if plaintext[pos1] == plaintext[pos2]:
                pt.append(self.alphabet.index(b'x'))
                shift2 += 1
            else:
                pt.append(plaintext[pos2])
        ct = super(SeriatedPlayfair, self).encrypt(pt, key[0])
        for i in range(0, int(len(ct) / key[1]), 2):
            for j in range(key[1]):
                if i * key[1] + j * 2 >= len(ct):
                    break
                ciphertext.append(ct[i * key[1] + j * 2])
                if i * key[1] + j * 2 >= len(ct):
                    break
            for j in range(key[1]):
                if i * key[1] + j * 2 + 1 >= len(ct):
                    break
                ciphertext.append(ct[i * key[1] + j * 2 + 1])
                if i * key[1] + j * 2 + 1 >= len(ct):
                    break
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        ct = [0]*len(ciphertext)
        cntr = 0
        for i in range(0, int(len(ciphertext) / key[1]), 2):
            for j in range(key[1]):
                if i * key[1] + j * 2 >= len(ciphertext):
                    break
                ct[i * key[1] + j * 2] = ciphertext[cntr]
                cntr += 1
                if i * key[1] + j * 2 >= len(ciphertext):
                    break
            for j in range(key[1]):
                if i * key[1] + j * 2 + 1 >= len(ciphertext):
                    break
                ct[i * key[1] + j * 2 + 1] = ciphertext[cntr]
                cntr += 1
                if i * key[1] + j * 2 + 1 >= len(ciphertext):
                    break
        pt = super(SeriatedPlayfair, self).decrypt(ct, key[0])
        plaintext = [0]*len(pt)
        shift1 = 0
        shift2 = 0
        cntr = 0
        for i in range(len(ciphertext) - key[1] * 3):
            if i % key[1] == 0 and shift1 != shift2:
                shift1 = shift2
                shift2 = 0
            pos = int(i / key[1]) * key[1] * 2 + i % key[1]
            pos1 = pos - shift1
            pos2 = pos + key[1] - shift2 - shift1
            if cntr >= len(pt):
                break
            plaintext[pos1] = pt[cntr]
            cntr += 1
            if cntr >= len(pt):
                break
            if pt[cntr] == self.alphabet.index(b'x') and pt[cntr-1] == pt[cntr+2]:
                shift2 += 1
                plaintext = plaintext[:-1]
            plaintext[pos2] = pt[cntr]
            cntr += 1
        return np.array(plaintext)

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower().replace(b'j', b'i')
        if not keep_unknown_symbols:
            return remove_unknown_symbols(plaintext, self.alphabet)
        return plaintext
