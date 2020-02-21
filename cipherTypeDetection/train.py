import os
import tensorflow as tf

plaintext_files = []
for name in dir:
    if os.path.isfile(os.path.join(args.input_folder, name)):
        plaintext_files.append(os.path.join(args.input_folder, name))
dataset = tf.data.TextLineDataset(plaintext_files, num_parallel_reads=args.dataset_worksers)