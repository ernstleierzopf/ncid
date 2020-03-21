from cipherImplementations.simpleSubstitution import SimpleSubstitution
from cipherImplementations.vigenere import Vigenere
from cipherImplementations.columnarTransposition import ColumnarTransposition
from cipherImplementations.playfair import Playfair
from cipherImplementations.hill import Hill

ALPHABET = b'abcdefghijklmnopqrstuvwxyz'
UNKNOWN_SYMBOL = b'?'
UNKNOWN_SYMBOL_NUMBER = 90

CIPHER_TYPES = ['simple_substitution', 'vigenere', 'columnar_transposition', 'playfair', 'hill']
CIPHER_IMPLEMENTATIONS = [SimpleSubstitution(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Vigenere(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          ColumnarTransposition(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Playfair(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Hill(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER)]
KEY_LENGTHS = [[None]*13, list(range(4, 17)), list(range(4, 17)), list(range(4, 17)), [None]*13]
MTC3 = 'mtc3'