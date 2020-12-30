from cipherImplementations.railfence import Railfence
from unit.cipherImplementations.CipherTestBase import CipherTestBase
import numpy as np


class RailfenceTest(CipherTestBase):
    cipher = Railfence(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'Civil war field cipher.'
    ciphertext = b'clfdhiiwrilcpevaeir'
    decrypted_plaintext = b'civilwarfieldcipher'
    key = [np.array([0,1,2]), 0]

    # ciphertext = b'iipvlfeihiwrlcecadr'
    # key = [np.array([0,1,2,3]), 3]

    def test1generate_random_key(self):
        length = 5
        key, offset = self.cipher.generate_random_key(length)
        self.assertTrue(np.array_equal(key, np.array(list(range(length)))))
        self.assertTrue(0 <= offset <= 15)

        length = 19
        key, offset = self.cipher.generate_random_key(length)
        self.assertTrue(np.array_equal(key, np.array(list(range(length)))))
        self.assertTrue(0 <= offset <= 15)

        length = 25
        key, offset = self.cipher.generate_random_key(length)
        self.assertTrue(np.array_equal(key, np.array(list(range(length)))))
        self.assertTrue(0 <= offset <= 15)

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()
