from cipherImplementations.slidefair import Slidefair
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.utils import map_text_into_numberspace


class SlidefairTest(CipherTestBase):
    cipher = Slidefair(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'The Slidefair can be used with Vigenere , Variant or Beaufort.'
    ciphertext = b'ewkmcrnuafcxtjyqmmyyfutigwzpkhjmpkbsaieckvcfmiilci'
    decrypted_plaintext = b'theslidefaircanbeusedwithvigenerevariantorbeaufort'
    key = map_text_into_numberspace(b'digraph', cipher.alphabet, cipher.unknown_symbol_number)

    def test1generate_random_key(self):
        length = 5
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for c in key:
            self.assertIn(c, self.cipher.alphabet)

        length = 19
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for c in key:
            self.assertIn(c, self.cipher.alphabet)

        length = 25
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
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
