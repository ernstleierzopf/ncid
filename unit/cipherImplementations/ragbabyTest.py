from cipherImplementations.ragbaby import Ragbaby
from util.utils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class RagbabyTest(CipherTestBase):
    cipher = Ragbaby(CipherTestBase.ALPHABET.replace(b'j', b'').replace(b'x', b'') + b' ', CipherTestBase.UNKNOWN_SYMBOL,
                     CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'Word divisions are kept'
    ciphertext = b'ybbl hngqdufgl def hfyr'
    decrypted_plaintext = b'word divisions are kept'
    key = map_text_into_numberspace(b'grosbeakcdfhilmnpqtuvwyz', cipher.alphabet, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)

    def test1generate_random_key(self):
        length = 5
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        for c in key:
            self.assertIn(c, self.cipher.alphabet)

        length = 19
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        for c in key:
            self.assertIn(c, self.cipher.alphabet)

        length = 25
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        for c in key:
            self.assertIn(c, self.cipher.alphabet)

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
