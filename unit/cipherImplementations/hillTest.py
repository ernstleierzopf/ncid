from cipherImplementations.hill import Hill
from unit.cipherImplementations.CipherTestBase import CipherTestBase
import numpy as np


class HillTest(CipherTestBase):
    UNKNOWN_SYMBOL = b'x'
    UNKNOWN_SYMBOL_NUMBER = ord('x')
    cipher = Hill(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'this is a plaintext with special characters!%%xy<d'
    ciphertext = b'jufdtmdkdtheluizfpenrzzherhtyyvbmropizoz'
    decrypted_plaintext = b'thisisaplaintextwithspecialcharactersxyd'
    key = np.array([[2,15,22,3], [1,9,1,12], [16,7,13,11], [8,5,9,6]])

    def test1generate_random_key(self):
        for _ in range(0, 10):
            key = self.cipher.generate_random_key()
            self.assertEqual(4, len(key))
            for arr in key:
                self.assertEqual(4, len(arr))

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()