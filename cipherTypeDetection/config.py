from cipherImplementations.simpleSubstitution import SimpleSubstitution
from cipherImplementations.vigenere import Vigenere
from cipherImplementations.columnarTransposition import ColumnarTransposition
from cipherImplementations.playfair import Playfair
from cipherImplementations.hill import Hill
from cipherImplementations.amsco import Amsco
from cipherImplementations.autokey import Autokey
from cipherImplementations.baconian import Baconian
from cipherImplementations.bazeries import Bazeries
from cipherImplementations.beaufort import Beaufort
from cipherImplementations.bifid import Bifid
from cipherImplementations.cadenus import Cadenus
from cipherImplementations.checkerboard import Checkerboard
from cipherImplementations.condi import Condi
from cipherImplementations.cmbifid import CMBifid
from cipherImplementations.digrafid import Digrafid
from cipherImplementations.foursquare import Foursquare
from cipherImplementations.fractionedMorse import FractionedMorse
from cipherImplementations.grandpre import Grandpre
from cipherImplementations.grille import Grille
from cipherImplementations.gromark import Gromark
from cipherImplementations.gronsfeld import Gronsfeld
from cipherImplementations.headlines import Headlines
from cipherImplementations.homophonic import Homophonic
from cipherImplementations.monomeDinome import MonomeDinome
from cipherImplementations.morbit import Morbit

ALPHABET = b'abcdefghijklmnopqrstuvwxyz'
UNKNOWN_SYMBOL = b'?'
UNKNOWN_SYMBOL_NUMBER = 90

# CIPHER_TYPES = ['columnar_transposition', 'hill', 'playfair', 'simple_substitution', 'vigenere']
CIPHER_TYPES = ['amsco', 'autokey', 'baconian', 'bazeries', 'beaufort', 'bifid', 'cadenus', 'checkboard', 'columnar_transposition', 'condi',
                'cmbifid', 'digrafid', 'foursquare', 'fractioned_morse', 'grandpre', 'grille', 'gromark', 'gronsfeld', 'headlines',
                'homophonic',  # , 'incomplete_columnar_transposition'
                'monome_dinome', 'morbit']
# CIPHER_IMPLEMENTATIONS = [ColumnarTransposition(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
#                           Hill(ALPHABET, b'x', ord('x')),
#                           Playfair(ALPHABET.replace(b'j', b''), b'x', ord('x')),
#                           SimpleSubstitution(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
#                           Vigenere(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER)]
CIPHER_IMPLEMENTATIONS = [Amsco(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Autokey(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Baconian(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Bazeries(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Beaufort(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Bifid(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Cadenus(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Checkerboard(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          ColumnarTransposition(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER, fill_blocks=True),
                          Condi(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          CMBifid(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Digrafid(ALPHABET + b'#', UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Foursquare(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          FractionedMorse(ALPHABET + b' ', UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Grandpre(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Grille(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Gromark(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Gronsfeld(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Headlines(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Homophonic(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          # incomplete Columnar Transposition: how is it differentiable to the normal columnar transposition?
                          # ColumnarTransposition(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER, fill_blocks=True)
                          MonomeDinome(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Morbit(ALPHABET + b' ', UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER)]

# KEY_LENGTHS = [13, None, 13, None, 13]
# KEY_LENGTHS = [list(range(4, 17)), [None]*13, list(range(4, 17)), [None]*13, list(range(4, 17))]
# KEY_LENGTHS = [[5,10,20,25], [None]*4, [6,7,8,9], [None]*4, [5,10,20,25]]
# KEY_LENGTHS = [[None]*4, [5,10,20,25]]
KEY_LENGTHS = [[5,6,7,8], [5,6,7,8], [None]*4, [None]*4, [5,6,7,8], [5,6,7,8], [4,4,4,4], [5,10,15,20], [5,6,7,8], [None]*4, [5,6,7,8],
               [5,6,7,8], [None]*4, [None]*4, [None]*4, [2,5,10,5], [None]*4, [5,6,7,8], [None]*4, [None]*4, [None]*4, [None]*4]
MTC3 = 'mtc3'
ACA = 'aca'