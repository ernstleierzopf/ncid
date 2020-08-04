from cipherImplementations.pollux import Pollux
from util.textUtils import map_text_into_numberspace, map_numbers_into_textspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase
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
    ciphertext = [0,8,6,3,9,3,4,2,5,7,0,2,4,1,7,6,8,5,9,6,3,0,4,1,4,5,6,2,3,4,9,0,8,7,4,5,3,6,0]
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
        cntr = 0
        for ct in self.ciphertext:
            self.assertEqual(self.cipher.key_morse[ct], self.cipher.key_morse[ciphertext_numbers[cntr]])
            cntr += 1

    def test6decrypt(self):
        ciphertext = b'aigdjdecfhacebhgifjgdaebefgcdejaihefdga'
        ciphertext_numbers = map_text_into_numberspace(ciphertext, self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        plaintext_numbers = self.cipher.decrypt(ciphertext_numbers, self.key)
        plaintext = map_numbers_into_textspace(plaintext_numbers, self.cipher.alphabet, self.UNKNOWN_SYMBOL)
        self.assertEqual(self.decrypted_plaintext, plaintext)