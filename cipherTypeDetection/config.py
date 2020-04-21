from cipherImplementations.simpleSubstitution import SimpleSubstitution
from cipherImplementations.vigenere import Vigenere
from cipherImplementations.columnarTransposition import ColumnarTransposition
from cipherImplementations.playfair import Playfair
from cipherImplementations.hill import Hill

ALPHABET = b'abcdefghijklmnopqrstuvwxyz'
UNKNOWN_SYMBOL = b'?'
UNKNOWN_SYMBOL_NUMBER = 90

CIPHER_TYPES = ['columnar_transposition', 'hill', 'playfair', 'simple_substitution', 'vigenere']
# CIPHER_TYPES = ['hill', 'vigenere']
CIPHER_IMPLEMENTATIONS = [ColumnarTransposition(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Hill(ALPHABET, b'x', ord('x')),
                          Playfair(ALPHABET.replace(b'j', b''), b'x', ord('x')),
                          SimpleSubstitution(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Vigenere(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER)]
# CIPHER_IMPLEMENTATIONS = [Hill(ALPHABET, b'x', ord('x')),
#                           Vigenere(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER)]

# KEY_LENGTHS = [13, None, 13, None, 13]
# KEY_LENGTHS = [list(range(4, 17)), [None]*13, list(range(4, 17)), [None]*13, list(range(4, 17))]
KEY_LENGTHS = [[5,10,20,25], [None]*4, [6,7,8,9], [None]*4, [5,10,20,25]]
# KEY_LENGTHS = [[None]*4, [5,10,20,25]]
MTC3 = 'mtc3'
