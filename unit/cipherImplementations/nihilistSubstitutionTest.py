from cipherImplementations.nihilistSubstitution import NihilistSubstitution
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.textUtils import map_text_into_numberspace


class NihilistSubstitutionTest(CipherTestBase):
    cipher = NihilistSubstitution(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'The early bird'
    key = [map_text_into_numberspace(b'easy', cipher.alphabet, cipher.unknown_symbol_number),
           map_text_into_numberspace(b'simpleabcdfghknoqrtuvwxyz', cipher.alphabet, cipher.unknown_symbol_number)]
    # ct = [65, 55, 32, 75, 43, 65, 26, 8, 44, 34, 54, 79]
    # ciphertext = b''
    # for c in ct:
    #     ciphertext += bytes([CipherTestBase.ALPHABET[int(c / 10)]])
    #     ciphertext += bytes([CipherTestBase.ALPHABET[int(c % 10)]])
    # print(ciphertext)
    ciphertext = b'gfffdchfedgfcgaieedefehj'
    decrypted_plaintext = b'theearlybird'

    def test1generate_random_key_allowed_length(self):
        length = 5
        for _ in range(0, 100):
            keyword, key = self.cipher.generate_random_key(length)
            self.assertEqual(len(key), len(self.cipher.alphabet))
            self.assertEqual(len(keyword), length)

        length = 17
        keyword, key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        self.assertEqual(len(keyword), length)
        for _ in range(0, 100):
            keyword, key = self.cipher.generate_random_key(length)
            self.assertEqual(len(key), len(self.cipher.alphabet))
            self.assertEqual(len(keyword), length)

        length = 25
        keyword, key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        self.assertEqual(len(keyword), length)
        for _ in range(0, 100):
            keyword, key = self.cipher.generate_random_key(length)
            self.assertEqual(len(key), len(self.cipher.alphabet))
            self.assertEqual(len(keyword), length)

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.assertEqual(self.cipher.filter(self.plaintext, keep_unknown_symbols=False), self.decrypted_plaintext.replace(b'x', b''))

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()