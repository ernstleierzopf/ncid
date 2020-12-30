from cipherImplementations.cipher import Cipher, generate_keyword_alphabet, generate_random_keyword, OUTPUT_ALPHABET
from cipherImplementations.columnarTransposition import ColumnarTransposition
import numpy as np


def arrange_table(size,  text, horizontal_vertical_flag):
    split_size = int(len(text) / size + 1)
    if horizontal_vertical_flag:
        # horizontal table
        table = [[] for _ in range(size)]
        i = 0
        for i, c in enumerate(text):
            table[int(i / split_size)].append(c)
        table[int(i / split_size)].append(OUTPUT_ALPHABET.index(b'#'))
    else:
        # vertical table
        table = [[] for _ in range(split_size)]
        i = 0
        for i, c in enumerate(text):
            table[int(i / size)].append(c)
        table[int(i / size)].append(OUTPUT_ALPHABET.index(b'#'))
    return table


class Digrafid(Cipher):
    """
    Decryption currently only works for maximally 1 remaining table of numbers!! This cipher should be tested again. Also the # must be
    filtered to be able to calculate features.
    """

    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number
        self.col_trans = ColumnarTransposition(alphabet, self.unknown_symbol, self.unknown_symbol_number, False)

    def generate_random_key(self, length):
        alphabet = self.alphabet.replace(b'#', b'')
        key1 = generate_keyword_alphabet(alphabet, generate_random_keyword(alphabet, length))
        key2 = generate_keyword_alphabet(alphabet, generate_random_keyword(alphabet, length), vertical=True)
        return [length, key1, key2]

    def encrypt(self, plaintext, key):
        if len(plaintext) % 2 != 0:
            raise AttributeError('The Digrafid cipher needs an even length of plaintext.')
        horizontal_table = arrange_table(3, key[1], True)
        vertical_table = arrange_table(3, key[2], False)
        ciphertext_table = [[] for _ in range(3)]
        for i in range(0, len(plaintext), 2):
            j = 0
            while plaintext[i] not in horizontal_table[j]:
                j += 1
            ciphertext_table[0].append(horizontal_table[j].index(plaintext[i]))
            k = 0
            while plaintext[i + 1] not in vertical_table[k]:
                k += 1
            ciphertext_table[1].append(j * 3 + vertical_table[k].index(plaintext[i + 1]))
            ciphertext_table[2].append(k)
        ciphertext = []

        rest = (int(len(plaintext) / 2) * 3) % (key[0] * 3)
        for i in range(0, int(len(plaintext) / 2) * 3 - rest, 3):
            fraction = int(i / 3 / key[0])
            horizontal = ciphertext_table[int(i / key[0]) % 3][fraction * key[0] + i % key[0]]
            pos = ciphertext_table[int((i + 1) / key[0]) % 3][fraction * key[0] + (i + 1) % key[0]]
            vertical = ciphertext_table[int((i + 2) / key[0]) % 3][fraction * key[0] + (i + 2) % key[0]]
            ciphertext.append(horizontal_table[int(pos / 3)][horizontal])
            ciphertext.append(vertical_table[vertical][pos % 3])

        # add remainers
        shift = 0
        add = 0
        for i in range(int(len(plaintext) / 2) * 3 - rest, int(len(plaintext) / 2) * 3, 3):
            fraction = int(i / 3 / key[0])
            if fraction * key[0] + add % key[0] == len(ciphertext_table[int(add / key[0]) % 3]):
                shift += 1
                add = 0
            horizontal = ciphertext_table[int(add / key[0] + shift) % 3][fraction * key[0] + add % key[0]]
            add += 1
            if fraction * key[0] + add % key[0] == len(ciphertext_table[int(add / key[0]) % 3]):
                shift += 1
                add = 0
            pos = ciphertext_table[int(add / key[0] + shift) % 3][fraction * key[0] + add % key[0]]
            add += 1
            if fraction * key[0] + add % key[0] == len(ciphertext_table[int(add / key[0]) % 3]):
                shift += 1
                add = 0
            vertical = ciphertext_table[int(add / key[0] + shift) % 3][fraction * key[0] + add % key[0]]
            add += 1
            ciphertext.append(horizontal_table[int(pos / 3)][horizontal])
            ciphertext.append(vertical_table[vertical][pos % 3])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        horizontal_table = arrange_table(3, key[1], True)
        vertical_table = arrange_table(3, key[2], False)
        plaintext_table = [[] for _ in range(3)]
        cntr = 0
        rest = int((int(len(ciphertext) / 2) * 3) % (key[0] * 3) * 2 / 3)
        for i in range(0, len(ciphertext) - rest, 2):
            j = 0
            while ciphertext[i] not in horizontal_table[j]:
                j += 1
            plaintext_table[int(cntr / key[0]) % 3].append(horizontal_table[j].index(ciphertext[i]))
            cntr += 1
            k = 0
            while ciphertext[i + 1] not in vertical_table[k]:
                k += 1
            plaintext_table[int(cntr / key[0]) % 3].append(j * 3 + vertical_table[k].index(ciphertext[i + 1]))
            cntr += 1
            plaintext_table[int(cntr / key[0]) % 3].append(k)
            cntr += 1

        # add remainers
        cntr = 0
        for i in range(len(ciphertext) - rest, len(ciphertext), 2):
            j = 0
            while ciphertext[i] not in horizontal_table[j]:
                j += 1
            plaintext_table[int(cntr / rest) % 3].append(horizontal_table[j].index(ciphertext[i]))
            cntr += 2
            k = 0
            while ciphertext[i + 1] not in vertical_table[k]:
                k += 1
            plaintext_table[int(cntr / rest) % 3].append(j * 3 + vertical_table[k].index(ciphertext[i + 1]))
            cntr += 2
            plaintext_table[int(cntr / rest) % 3].append(k)
            cntr += 2
        plaintext = []
        for i in range(len(plaintext_table[0])):
            horizontal = plaintext_table[0][i]
            pos = plaintext_table[1][i]
            vertical = plaintext_table[2][i]
            plaintext.append(horizontal_table[int(pos / 3)][horizontal])
            plaintext.append(vertical_table[vertical][pos % 3])
        return np.array(plaintext)
