from cipherImplementations.cadenus import Cadenus
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.textUtils import map_text_into_numberspace


class CadenusTest(CipherTestBase):
    cipher = Cadenus(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'A severe limitation on the usefulness of the Cadenus is that every message must be a multiple of twenty-five letters long'
    ciphertext = b'systretomtattlusoatleeesfiyheasdfnmschbhneuvsnpmtofarenuseieeieltarlmentieetogevesitfaisltngeeuvowul'
    decrypted_plaintext = b'aseverelimitationontheusefulnessofthecadenusisthateverymessagemustbeamultipleoftwentyfiveletterslong'
    key = [map_text_into_numberspace(b'easy', CipherTestBase.ALPHABET, cipher.unknown_symbol_number),
           map_text_into_numberspace(b'azyxvutsrqponmlkjihgfedcb', CipherTestBase.ALPHABET, cipher.unknown_symbol_number)]

    def test1generate_random_key_allowed_length(self):
        length = 5
        keyword, key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet) - 1)
        self.assertEqual(len(keyword), length)
        self.assertEqual(key, b'azyxvutsrqponmlkjihgfedcb')
        for _ in range(0, 100):
            keyword, key = self.cipher.generate_random_key(length)
            self.assertEqual(len(key), len(self.cipher.alphabet) - 1)
            self.assertEqual(len(keyword), length)
            self.assertNotIn(b'w', key)

        length = 17
        keyword, key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet) - 1)
        self.assertEqual(len(keyword), length)
        self.assertEqual(key, b'azyxvutsrqponmlkjihgfedcb')
        for _ in range(0, 100):
            keyword, key = self.cipher.generate_random_key(length)
            self.assertEqual(len(key), len(self.cipher.alphabet) - 1)
            self.assertEqual(len(keyword), length)
            self.assertNotIn(b'w', key)

        length = 25
        keyword, key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet) - 1)
        self.assertEqual(len(keyword), length)
        self.assertEqual(key, b'azyxvutsrqponmlkjihgfedcb')
        for _ in range(0, 100):
            keyword, key = self.cipher.generate_random_key(length)
            self.assertEqual(len(key), len(self.cipher.alphabet) - 1)
            self.assertEqual(len(keyword), length)
            self.assertNotIn(b'w', key)

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()
        self.assertRaises(ValueError, self.cipher.generate_random_key, 26)

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.assertRaises(ValueError, self.cipher.encrypt, map_text_into_numberspace(b'thisissometextwhichisnotdivisiblebytwentyfive',
                          self.cipher.alphabet, self.cipher.unknown_symbol_number), [map_text_into_numberspace(
                              b'i', self.cipher.alphabet, self.cipher.unknown_symbol_number), map_text_into_numberspace(
                              b'azyxvutsrqponmlkjihgfedcb', self.cipher.alphabet, self.cipher.unknown_symbol_number)])
        self.assertRaises(ValueError, self.cipher.encrypt, map_text_into_numberspace(b'thisissometextwhichisnotdivisiblebytwentyfive',
                          self.cipher.alphabet, self.cipher.unknown_symbol_number), [map_text_into_numberspace(
                              b'i', self.cipher.alphabet, self.cipher.unknown_symbol_number), map_text_into_numberspace(
                              b'azyxvutsrqponmlkjihgfedcb', self.cipher.alphabet, self.cipher.unknown_symbol_number)])
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()
