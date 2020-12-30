from cipherImplementations.pollux import Pollux
from cipherImplementations.cipher import OUTPUT_ALPHABET
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.textUtils import map_text_into_numberspace
import numpy as np


class PolluxTest(CipherTestBase):
    cipher = Pollux(CipherTestBase.ALPHABET + b' ', CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'Luck helps.'
    # ciphertext = b'086393425702417685963041456234908745360'
    # ct = []
    # for c in ciphertext:
    #     ct.append(int(bytes([c])))
    # ciphertext = map_numbers_into_textspace(ct, cipher.alphabet, cipher.unknown_symbol)
    # print(ciphertext)
    ciphertext = b'086393425702417685963041456234908745360'
    decrypted_plaintext = b'luck helps'
    key = np.array([0,1,2,3,4,5,6,7,8,9])

    def test1generate_random_key(self):
        for _ in range(0, 100):
            key = self.cipher.generate_random_key()
            self.assertEqual(10, len(key))

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=False)
        plaintext_numbers = map_text_into_numberspace(plaintext, self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key)
        ciphertext_numbers = np.array([int(bytes([OUTPUT_ALPHABET[c]])) for c in ciphertext_numbers])
        cntr = 0
        for ct in self.ciphertext:
            ct = int(bytes([ct]))
            self.assertEqual(self.cipher.key_morse[ct], self.cipher.key_morse[ciphertext_numbers[cntr]])
            cntr += 1

    def test6decrypt(self):
        self.run_test6decrypt()
