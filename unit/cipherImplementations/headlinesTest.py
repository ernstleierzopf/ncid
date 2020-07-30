from cipherImplementations.headlines import Headlines
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.textUtils import map_text_into_numberspace


class HeadlinesTest(CipherTestBase):
    cipher = Headlines(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    # these values differ from the description as the plaintext is split up into 5 equally sized plaintexts and the corresponding setting
    # is used for every part.
    plaintext = b'Bush Signs Intelligence Overhaul Legislation Bin Laden Urges Fighters to Strike Oil Facilities Pfizer: Painkiller may' \
                b' pose increased cardiovascular risk Carrey masters disguises in Lemony Snicket Martinez blasts ex-teammate Schilli'
    ciphertext = b'gctjtnwotnoalzznwlodlphlxjfczzlwntzfanpovzsiuxysfdhyoczhwpydoptopdzmytzicurzizpzwzoaylwfotyidyeewfxtjonzwyiufwtzwcutf' \
                 b'cynzqdgbuqhhfdegqhhpayqdkphdwfdnbfdpdfrupyxwbjwgsaqyuioygwqhcvijyjqpyqiuuiyqjsxgvvg'
    decrypted_plaintext = b'bushsignsintelligenceoverhaullegislationbinladenurgesfighterstostrikeoilfacilitiespfizerpainkillermayposeinc' \
                          b'reasedcardiovascularriskcarreymastersdisguisesinlemonysnicketmartinezblastsexteammateschilli'
    key = [map_text_into_numberspace(b'drugs', CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL_NUMBER),
           map_text_into_numberspace(b'cfuaptosnzilyejwhgvbqmkxdr', CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)]

    def test1generate_random_key(self):
        old_key = self.cipher.alphabet
        old_setting = b'drugs'
        for _ in range(0, 100):
            setting, key = self.cipher.generate_random_key()
            self.assertEqual(26, len(key))
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