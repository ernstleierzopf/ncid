from cipherImplementations.checkerboard import Checkerboard
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.textUtils import map_text_into_numberspace, map_numbers_into_textspace
import copy


class CheckerboardTest(CipherTestBase):
    cipher = Checkerboard(CipherTestBase.ALPHABET.replace(b'j', b''), CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'numberscanalsobeusedascoordinates'
    ciphertext = [b'hrrysgsseaoaoyesrshrrsegoyrgssearyoyeaeyrsoyesrgrgoaeyhahrrsoseaoy',
                  b'bhatcwcekililtkeaebhaekwltawcekiatltkiktaeltkeawawliktbibhaelekilt']
    decrypted_plaintext = b'numberscanalsobeusedascoordinates'
    key = [map_text_into_numberspace(b'horseblack', cipher.alphabet, cipher.unknown_symbol_number), map_text_into_numberspace(
           b'grayswhite', cipher.alphabet, cipher.unknown_symbol_number), map_text_into_numberspace(
        b'knighpqrstoyzuamxwvblfedc', cipher.alphabet, cipher.unknown_symbol_number)]

    def test1generate_random_key_allowed_length(self):
        length = 5
        rowkey, columnkey, alphabet = self.cipher.generate_random_key(length)
        self.assertEqual(len(rowkey), length)
        self.assertEqual(len(columnkey), length)
        alph = copy.copy(self.cipher.alphabet)
        for c in rowkey:
            self.assertIn(c, alph)
            alph = alph.replace(bytes([c]), b'')
        alph = copy.copy(self.cipher.alphabet)
        for c in columnkey:
            self.assertIn(c, alph)
            alph = alph.replace(bytes([c]), b'')
        alph = copy.copy(self.cipher.alphabet)
        for c in alphabet:
            self.assertIn(c, alph)
            alph = alph.replace(bytes([c]), b'')
        self.assertEqual(alph, b'')

        length = 10
        rowkey, columnkey, alphabet = self.cipher.generate_random_key(length)
        self.assertEqual(len(rowkey), length)
        self.assertEqual(len(columnkey), length)
        alph = copy.copy(self.ALPHABET)
        for c in rowkey:
            self.assertIn(c, alph)
            alph = alph.replace(bytes([c]), b'')
        alph = copy.copy(self.ALPHABET)
        for c in columnkey:
            self.assertIn(c, alph)
            alph = alph.replace(bytes([c]), b'')
        alph = copy.copy(self.cipher.alphabet)
        for c in alphabet:
            self.assertIn(c, alph)
            alph = alph.replace(bytes([c]), b'')
        self.assertEqual(alph, b'')

        length = 25
        rowkey, columnkey, alphabet = self.cipher.generate_random_key(length)
        self.assertEqual(len(rowkey), length)
        self.assertEqual(len(columnkey), length)
        alph = copy.copy(self.ALPHABET)
        for c in rowkey:
            self.assertIn(c, alph)
            alph = alph.replace(bytes([c]), b'')
        alph = copy.copy(self.ALPHABET)
        for c in columnkey:
            self.assertIn(c, alph)
            alph = alph.replace(bytes([c]), b'')
        alph = copy.copy(self.cipher.alphabet)
        for c in alphabet:
            self.assertIn(c, alph)
            alph = alph.replace(bytes([c]), b'')
        self.assertEqual(alph, b'')

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=False)
        plaintext_numbers = map_text_into_numberspace(plaintext, self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key)
        ciphertext = map_numbers_into_textspace(ciphertext_numbers, self.cipher.alphabet, self.UNKNOWN_SYMBOL)
        for i, c in enumerate(ciphertext):
            found = False
            for ct in self.ciphertext:
                if ct[i] == c:
                    found = True
                    break
            self.assertTrue(found, 'ciphertext: %s, i: %d, c: %s' % (bytes(ciphertext), i, bytes([c])))

    def test6decrypt(self):
        ciphertext_numbers = map_text_into_numberspace(self.ciphertext[0], self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        plaintext_numbers = self.cipher.decrypt(ciphertext_numbers, self.key)
        plaintext = map_numbers_into_textspace(plaintext_numbers, self.cipher.alphabet, self.UNKNOWN_SYMBOL)
        self.assertEqual(self.decrypted_plaintext, plaintext)
