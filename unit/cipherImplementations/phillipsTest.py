from cipherImplementations.phillips import Phillips
from util.textUtils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class PhillipsTest(CipherTestBase):
    cipher = Phillips(CipherTestBase.ALPHABET.replace(b'j', b''), CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'Squares one and five are actually the same as are squares two and eight. The overall period is forty.'
    ciphertext = b'kzwlytgedtqetarbtygtlfxwlppoxltykutkgkytkzwlytgxseqetirzqaaqtcitykppvblhefhgreyxo'
    decrypted_plaintext = b'squaresoneandfiveareactuallythesameasaresquarestwoandeighttheoverallperiodisforty'
    key = map_text_into_numberspace(b'diagocbslnefhkmutrqpvwxyz', cipher.alphabet, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)

    def test1generate_random_key(self):
        length = 5
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        for c in key:
            self.assertIn(c, self.cipher.alphabet)

        length = 19
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        for c in key:
            self.assertIn(c, self.cipher.alphabet)

        length = 25
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
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
