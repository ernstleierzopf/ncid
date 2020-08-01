from cipherImplementations.gromark import Gromark
from util.textUtils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class GromarkTest(CipherTestBase):
    cipher = Gromark(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'thereareuptotensubstitutesperletter'
    ciphertext = b'nfyckbtijcnwzycacjnaynlqpwwstwpjqfl'
    decrypted_plaintext = b'thereareuptotensubstitutesperletter'
    key = [[2,3,4,5,2], map_text_into_numberspace(b'ajrxebksygfpvidoumhqwncltz', CipherTestBase.ALPHABET,
           CipherTestBase.UNKNOWN_SYMBOL_NUMBER)]

    def test1generate_random_key(self):
        old_key = self.cipher.alphabet
        old_primer = [2,3,4,5,2]
        for i in range(3, 25):
            primer, key = self.cipher.generate_random_key(i)
            self.assertEqual(26, len(key))
            self.assertNotEqual(key, old_key)
            self.assertEqual(5, len(primer))
            self.assertNotEqual(primer, old_primer)
            for p in primer:
                self.assertIsInstance(p, int)
                self.assertTrue(p < 10)
            old_primer = primer
            old_key = key

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