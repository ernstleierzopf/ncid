import argparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='CANN Ciphertext Generator Script')
parser.add_argument('--input_folder', default='../../plaintexts', type=str,
                    help='Input folder of the plaintexts.')
parser.add_argument('--save_folder', default='../../mtc3_cipher_id',
                    help='Directory for saving generated ciphertexts.')
parser.add_argument('--append_key', default=False, type=str2bool,
                    help='Append the encryption key at the end of every line.')

args = parser.parse_args()
parser.set_defaults(keep_spaces=False)

#print all arguments for debugging..
for arg in vars(args):
    print("%s = %s"%(arg, vars(args)[arg]))