''' This is the interface of the cipher implementations.'''
class Cipher:
    def generate_random_key(self, alphabet, length):
        raise Exception('Interface method called')

    def encrypt (self, plaintext, key):
        raise Exception('Interface method called')

    def filter(self):
        raise Exception('Interface method called')