from cipherImplementations.playfair import Playfair
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.textUtils import map_text_into_numberspace, map_numbers_into_textspace


class PlayfairTest(CipherTestBase):
    cipher = Playfair(CipherTestBase.ALPHABET.replace(b'j', b''), b'x', ord('x'))
    plaintext = b'this is a plaintext with special characters!%xzll'
    ciphertext = b'xdnonoboickiudtutmxdoqfbodqfdyozavgpxycmvn'
    decrypted_plaintext = b'thisisaplaintextwithspecialcharactersxzlxl'
    key = map_text_into_numberspace(b'abczydefghiklmnopqrstuvwx', cipher.alphabet, cipher.unknown_symbol_number)

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

        length = 150
        self.assertRaises(ValueError, self.cipher.generate_random_key, length)

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()
        self.assertRaises(ValueError, self.cipher.generate_random_key, 27)

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt_remove_unknown_symbols(self):
        self.run_test5encrypt_remove_unknown_symbols()

    def test6decrypt_remove_unknown_symbols(self):
        self.run_test6decrypt_keep_unknown_symbols()