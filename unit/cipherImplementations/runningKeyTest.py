from cipherImplementations.runningKey import RunningKey
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class RunningKeyTest(CipherTestBase):
    cipher = RunningKey(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'This cipher can be used with any of the periodics.'
    ciphertext = b'bapspgdmxygprsmivmfo'
    decrypted_plaintext = b'thisciphercanbeusedwithanyoftheperiodics'
    key = None

    def test1generate_random_key(self):
        length = 5
        self.assertIsNone(self.cipher.generate_random_key(length))

        length = 19
        self.assertIsNone(self.cipher.generate_random_key(length))

        length = 25
        self.assertIsNone(self.cipher.generate_random_key(length))

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()

    # def test6decrypt(self):
    #     self.run_test6decrypt()
