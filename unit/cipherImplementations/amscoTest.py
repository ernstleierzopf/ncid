from cipherImplementations.amsco import Amsco
from unit.cipherImplementations.CipherTestBase import CipherTestBase
import random


class AmscoTest(CipherTestBase):
    cipher = Amsco(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'Incomplete columnar with alternating single letters and digraphs.'
    ciphertext = b'cecrteglenphplutnanteiomowirsitddsintnalinesaalemhatglrgr'
    decrypted_plaintext = b'incompletecolumnarwithalternatingsinglelettersanddigraphs'
    key = [4,1,3,2,5]

    def test1generate_random_key_allowed_length(self):
        for _ in range(0, 100):
            length = random.randint(2, 9)
            key = self.cipher.generate_random_key(length)
            self.assertEqual(length, len(key))

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