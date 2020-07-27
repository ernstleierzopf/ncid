from cipherImplementations.foursquare import Foursquare
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.textUtils import map_text_into_numberspace


class FoursquareTest(CipherTestBase):
    cipher = Foursquare(CipherTestBase.ALPHABET.replace(b'j', b''), CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'comequicklyweneedhelpx'
    ciphertext = b'lewixafnexcudxuvdpgxhz'
    decrypted_plaintext = b'comequicklyweneedhelpx'
    key = (b'grdlueyfnvoahpwmbiqxtcksz', b'licnvotdpwgheqxamfsyrbkuz')

    def test1generate_random_key_allowed_length(self):
        old_key1 = self.cipher.alphabet
        old_key2 = self.cipher.alphabet
        for _ in range(0, 100):
            key1, key2 = self.cipher.generate_random_key()
            self.assertEqual(25, len(key1))
            self.assertEqual(25, len(key2))
            self.assertNotEqual(key1, old_key1)
            self.assertNotEqual(key2, old_key2)
            self.assertNotEqual(key1, key2)
            old_key1 = key1
            old_key2 = key2

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()