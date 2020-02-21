from cipherImplementations import monoalphabetic_substitution

CIPHER_TYPES = ['monoalphabetic_substitution', 'vigenere', 'columnar_transposition', 'playfair', 'hill']
CIPHER_IMPLEMENTATIONS = [monoalphabetic_substitution.Monoalphabetic_substitution(b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', b'?', 90)]
KEY_LENGTHS = [None]
MTC3 = 'mtc3'