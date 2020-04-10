from cipherImplementations.simpleSubstitution import SimpleSubstitution
from cipherImplementations.vigenere import Vigenere
from cipherImplementations.columnarTransposition import ColumnarTransposition
from cipherImplementations.playfair import Playfair
from cipherImplementations.hill import Hill

ALPHABET = b'abcdefghijklmnopqrstuvwxyz'
UNKNOWN_SYMBOL = b'?'
UNKNOWN_SYMBOL_NUMBER = 90

CIPHER_TYPES = ['columnar_transposition', 'hill', 'playfair', 'simple_substitution', 'vigenere']
CIPHER_IMPLEMENTATIONS = [ColumnarTransposition(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Hill(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Playfair(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          SimpleSubstitution(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Vigenere(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER)]
#KEY_LENGTHS = [13, None, 13, None, 13]
KEY_LENGTHS = [list(range(4, 17)), [None]*13, list(range(4, 17)), [None]*13, list(range(4, 17))]
MTC3 = 'mtc3'