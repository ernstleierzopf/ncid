from unit.cipherImplementations.CipherTestBase import CipherTestBase
from cipherImplementations.cipher import Cipher


class CipherTest(CipherTestBase):

    def test1raiseException(self):
        self.assertRaises(Exception, Cipher)
