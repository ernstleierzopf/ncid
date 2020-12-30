from cipherImplementations.digrafid import Digrafid
from util.textUtils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class DigrafidTest(CipherTestBase):
    cipher = Digrafid(CipherTestBase.ALPHABET + b'#', CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'This is the forest pri'
    # period of cipher = 3
    ciphertext = b'hjmxwswjadwgfcspyi'
    decrypted_plaintext = b'thisistheforestpri'
    key = [3, map_text_into_numberspace(b'keywordabcfghijlmnpqstuvxz', cipher.alphabet, cipher.unknown_symbol_number),
           map_text_into_numberspace(b'vdpefqrgsthuijwckxamylnzbo', cipher.alphabet, cipher.unknown_symbol_number)]
    # period of cipher = 4
    # ciphertext = b'hjtkvhyuffwdsqypri'
    # key[0] = 4

    def test1generate_random_key(self):
        length = 1
        leng, key1, key2 = self.cipher.generate_random_key(length)
        self.assertEqual(len(key1), len(CipherTestBase.ALPHABET))
        self.assertEqual(len(key2), len(CipherTestBase.ALPHABET))
        self.assertEqual(leng, length)
        for c in key1:
            self.assertIn(c, self.cipher.alphabet)
        for c in key2:
            self.assertIn(c, self.cipher.alphabet)

        length = 3
        leng, key1, key2 = self.cipher.generate_random_key(length)
        self.assertEqual(len(key1), len(CipherTestBase.ALPHABET))
        self.assertEqual(len(key2), len(CipherTestBase.ALPHABET))
        self.assertEqual(leng, length)
        for c in key1:
            self.assertIn(c, self.cipher.alphabet)
        for c in key2:
            self.assertIn(c, self.cipher.alphabet)

        length = 19
        leng, key1, key2 = self.cipher.generate_random_key(length)
        self.assertEqual(len(key1), len(CipherTestBase.ALPHABET))
        self.assertEqual(len(key2), len(CipherTestBase.ALPHABET))
        self.assertEqual(leng, length)
        for c in key1:
            self.assertIn(c, self.cipher.alphabet)
        for c in key2:
            self.assertIn(c, self.cipher.alphabet)

        length = 150
        leng, key1, key2 = self.cipher.generate_random_key(length)
        self.assertEqual(len(key1), len(CipherTestBase.ALPHABET))
        self.assertEqual(len(key2), len(CipherTestBase.ALPHABET))
        self.assertEqual(leng, length)
        for c in key1:
            self.assertIn(c, self.cipher.alphabet)
        for c in key2:
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
