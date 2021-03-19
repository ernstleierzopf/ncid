from cipherImplementations.cmbifid import CMBifid
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.utils import map_text_into_numberspace


class CMBifidTest(CipherTestBase):
    cipher = CMBifid(CipherTestBase.ALPHABET.replace(b'j', b''), CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'Odd periods are popular.'
    ciphertext = b'fanxzexfenukkrbynkak'
    decrypted_plaintext = b'oddperiodsarepopular'
    key = [7, map_text_into_numberspace(b'extraklmpohwzqdgvusifcbyn', cipher.alphabet, cipher.unknown_symbol_number),
           map_text_into_numberspace(b'ncdrsobfquvagpweyhmxltikz', cipher.alphabet, cipher.unknown_symbol_number)]

    def test1generate_random_key_allowed_length(self):
        length = 5
        leng, key1, key2 = self.cipher.generate_random_key(length)
        self.assertEqual(len(key1), len(self.cipher.alphabet))
        self.assertEqual(len(key2), len(self.cipher.alphabet))
        self.assertEqual(leng, length)
        for c in key1:
            self.assertIn(c, self.cipher.alphabet)
        for c in key2:
            self.assertIn(c, self.cipher.alphabet)

        length = 19
        leng, key1, key2 = self.cipher.generate_random_key(length)
        self.assertEqual(len(key1), len(self.cipher.alphabet))
        self.assertEqual(len(key2), len(self.cipher.alphabet))
        self.assertEqual(leng, length)
        for c in key1:
            self.assertIn(c, self.cipher.alphabet)
        for c in key2:
            self.assertIn(c, self.cipher.alphabet)

        length = 150
        leng, key1, key2 = self.cipher.generate_random_key(length)
        self.assertEqual(len(key1), len(self.cipher.alphabet))
        self.assertEqual(len(key2), len(self.cipher.alphabet))
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
