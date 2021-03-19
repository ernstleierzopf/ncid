from cipherImplementations.seriatedPlayfair import SeriatedPlayfair
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.utils import map_text_into_numberspace


class SeriatedPlayfairTest(CipherTestBase):
    cipher = SeriatedPlayfair(
        CipherTestBase.ALPHABET.replace(b'j', b''), CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'Come quickly we need help immediately. tom.'
    ciphertext = b'nlbcspcdfgxzqqcdcmgcgqtbhcftrhfgwhgb'
    decrypted_plaintext = b'comequicklyweneedhelpimmediatelytom'
    key = [map_text_into_numberspace(b'logarithmbcdefknpqsuvwxyz', cipher.alphabet, cipher.unknown_symbol_number), 6]

    def test1generate_random_key_allowed_length(self):
        length = 5
        key, period = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        self.assertEqual(period, length)
        for c in key:
            self.assertTrue(c in self.ALPHABET)

        length = 19
        key, period = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        self.assertEqual(period, length)
        for c in key:
            self.assertTrue(c in self.ALPHABET)

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()

    def test4filter_delete_unknown_symbols(self):
        self.assertEqual(self.cipher.filter(self.plaintext, keep_unknown_symbols=False), self.decrypted_plaintext.replace(b'x', b''))

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()
