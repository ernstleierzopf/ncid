from cipherImplementations.headlines import Headlines
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.utils import map_text_into_numberspace


class HeadlinesTest(CipherTestBase):
    cipher = Headlines(CipherTestBase.ALPHABET + b' ', CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    # these values differ from the description as the plaintext is split up into 5 equally sized plaintexts and the corresponding setting
    # is used for every part.
    plaintext = b'Bush Signs Intelligence Overhaul Legislation Bin Laden Urges Fighters to Strike Oil Facilities Pfizer: Painkiller may' \
                b' pose increased cardiovascular risk Carrey masters disguises in Lemony Snicket Martinez blasts ex-teammate Schilli'
    ciphertext = b'gctj tnwot noalzznwlodl phlxjfcz zlwntzfanpo vzs iuxys fdhyo czhwpydo pt opdzmy tzi curzizpzwz oaylwf otyidyeewf xtj' \
                 b' onzw yiufwtzwc utfcynzqdgbuqh hfde gqhhpa yqdkphd wfdnbfdpd fr upyxwb jwgsaqy uioygwqh cvijyj qpyqiuuiyq jsxgvvg'
    decrypted_plaintext = b'bush signs intelligence overhaul legislation bin laden urges fighters to strike oil facilities pfizer' \
                          b' painkiller may pose increased cardiovascular risk carrey masters disguises in lemony snicket martinez blasts' \
                          b' exteammate schilli'
    key = [map_text_into_numberspace(b'drugs', CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL_NUMBER),
           map_text_into_numberspace(b'cfuaptosnzilyejwhgvbqmkxdr', CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)]

    def test1generate_random_key(self):
        old_key = self.cipher.alphabet
        old_setting = b'drugs'
        for i in range(1, 25):
            setting, key = self.cipher.generate_random_key(i)
            self.assertEqual(len(self.cipher.alphabet), len(key))
            self.assertNotEqual(key, old_key)
            self.assertEqual(5, len(setting))
            self.assertNotEqual(setting, old_setting)
            old_setting = setting
            old_key = key

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()
