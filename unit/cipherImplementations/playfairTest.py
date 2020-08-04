from cipherImplementations.playfair import Playfair
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.textUtils import map_text_into_numberspace


class PlayfairTest(CipherTestBase):
    cipher = Playfair(CipherTestBase.ALPHABET.replace(b'j', b''), CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'come quickly we need help'
    ciphertext = b'dlhfsncncrzxcqqgfeeqon'
    decrypted_plaintext = b'comequicklywenexedhelp'
    key = map_text_into_numberspace(b'logarithmbcdefknpqsuvwxyz', cipher.alphabet, cipher.unknown_symbol_number)

    def test1generate_random_key_allowed_length(self):
        length = 5
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        for c in key:
            self.assertTrue(c in self.ALPHABET)

        length = 19
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        for c in key:
            self.assertTrue(c in self.ALPHABET)

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()
        self.assertRaises(ValueError, self.cipher.generate_random_key, 26)

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()