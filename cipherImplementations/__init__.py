from cipherImplementations.simple_substitution import Simple_substitution
from cipherImplementations.vigenere import Vigenere
from cipherImplementations.columnar_transposition import Columnar_transposition
from cipherImplementations.playfair import Playfair
from cipherImplementations.hill import Hill

ALPHABET = b'abcdefghijklmnopqrstuvwxyz'
UNKNOWN_SYMBOL = b'?'
UNKNOWN_SYMBOL_NUMBER = 90

CIPHER_TYPES = ['simple_substitution', 'vigenere', 'columnar_transposition', 'playfair', 'hill']
CIPHER_IMPLEMENTATIONS = [Simple_substitution(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Vigenere(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Columnar_transposition(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Playfair(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Hill(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER)]
KEY_LENGTHS = [None, 7, 10, 10, None]
MTC3 = 'mtc3'