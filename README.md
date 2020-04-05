# cann

A neuronal network to detect and analyse ciphers from historical texts.

# CANN - Crypto Analysis with Neuronal Networks

This project contains code for the detection and classification of ciphers to classical algorithms by using a neuronal network. In Future other parts of the crypto analysis will be implemented.

# Installation

- Clone this repository and enter it:
  ```Shell
  git clone https://github.com/dITySoftware/cann
  cd cann
  ```

- Set up the environment with the following method:
   - Set up a Python3.7 or higher environment (e.g., using virtenv).
   
   - Install all needed packages:
     
     ```Shell
     pip3 install tensorflow tensorflow_datasets numpy scikit_learn
     ```

- Install the recommended and tested versions by using requirements.txt:

  ```
  pip3 install -r reqirements.txt
  ```

## Generate Plaintexts (Optional)

First of all this usage is not recommended as first option. Try running the `train.py` or `eval.py` script with the argument `--download_dataset=True`, if you only want to train or test on the filtered dataset. Optionally you can download the dataset on your own from [here](https://drive.google.com/open?id=1bF5sSVjxTxa3DB-P5wxn87nxWndRhK_V)

If you'd like to create your own plaintexts, you can use the `generatePlainTextFiles.py` script. Therefore you first need to download some texts, for example the Gutenberg Library. You can do that by using following command, which downloads all English e-books compressed with zip.  Note that this script can take a while and dumps about 14gb of files into `./data/gutenberg_en` and 5.3gb additionaly if you do not delete the `gutenberg_en.zip`.

```shell
wget -m -H -nd "http://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en" -e use_proxy=yes -e http_proxy=46.101.1.221:80 > /tmp/wget-log 2>&1
```

The `generatePlainTextFiles.py` script automatically unpacks the zips, with the parameter `--restructure_directory`.  Every line in a plaintext is seperated by a '\n', so be sure to save it in the right format or use the `generatePlainTextFiles.py` script to reformat all files from '\r\n' to '\n'. For further description read the help by using the `--help` parameter. Example usage:

```
python3 generate_plain_text_files.py --directory=../gutenberg_en --restructure_directory=true
```

## Generate Ciphertexts (Optional)

You might want to evaluate one or more models by using the same ciphertext files. This evaluation type is practical for most comparable evaluation of models. The `generate_cipher_text_files.py` script encrypts plaintext files to multiple ciphertext files. The naming convention is *fileName-cipherType-minLenXXX-maxLenXXX-keyLenXX.txt*. This script generates ciphertexts out of plaintexts. If a line is not long enough it is concatenated with the next line. If a line is too long it is sliced into max_text_len length. For further description read the help by using the `--help` parameter. Example usage:

```
python3 generate_cipher_text_files.py --min_text_len=100 --max_text_len=100 --max_files_count=100
```

# TODO: Evaluation

Here are our CANN models (released on April 5th, 2020) along with average loss / accuracy and the results on random data:

| Model                                                        | average  loss / accuracy | cipher \| accuracy rate matrix on 1 million lines of plaintexts |
| ------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------ |
| mtc3_model_100.h5 (Link)<br />batch_size = 16<br />ciphers = mtc3<br />keep_unknown_symbols = False<br />train_len = 100 \| 100<br />test_len = 100 \| 100 |                          | simple_substitution\| accuracy rate1 <br />vigenere\| <br />columnar_transposition\| <br />playfair\| <br />hill\| |
| mtc3_logistic_regression_baseline_50.h5 (Link)               |                          | simple_substitution\| accuracy rate1 <br />vigenere\| <br />columnar_transposition\| <br />playfair\| <br />hill\ |
|                                                              |                          |                                                              |

There are multiple ways to evaluate the model. First of all it is needed to put the corresponding weights file in the `./weights` directory and run one of the following commands:

- **benchmark** - Use this argument to create ciphertexts on the fly, like in training mode, and evaluate them with the model. This option is optimized for large throughput to test the model. Example usage:

  ```
  python3 eval.py --model=./weights/model.h5 --max_iter=1000000 benchmark --dataset_size=1040 --dataset_workers=10 --min_text_len=100 --max_text_len=100
  ```

- **evaluate** - Use this argument to evaluate cipher types for directories with ciphertext files in it. There are two *evaluation_modes*: 

  - *per_file* - every file is evaluated on it's own. The average of all evaluations of that file is the output. 

  - *summarized* - all files are evaluated and the average is printed at the end of the script. This mode is like *benchmark*, but is more reproducible and comparable, because the inputs are consistent.

  Example usage:

  ```
  python3 eval.py --model=./weights/model.h5 --max_iter=100000 evaluate --evaluation_mode=per_file --ciphertext_folder=../data/mtc3_cipher_id
  ```

- **single_line** - Use this argument to predict a single line of ciphertext. The difference of this command is, that in contrast to the other modes, the results are predicted without knowledge of the real cipher type. There are two types of data this command can process:

  - *ciphertext* - A single line of ciphertext to be predicted by the model. Example usage:

  ```
  python3 eval.py --model=./weights/model.h5 single_line --ciphertext=yingraobhoarhthosortmkicwhaslcbpirpocuedcfthcezvoryyrsrdyaffcleaetiaaeuhtyegeadsneanmatedbtrdndres
  ```

  - *file* - A file with mixed lines of ciphertext to be predicted line by line by the model. Example usage:

  ```
  python3 eval.py --model=./weights/model.h5 single_line --verbose=false --file=../data/mixed_ciphertexts_alphabetic_5lines_per_cipher.txt
  ```

To see all options of `eval.py`, run the `--help` or `-h` command.

```
python3 eval.py --help
```

# Training

By default we train ciphers of the MysteryTwister C3 Challenge "Cipher ID". The plaintexts I used are already filtered and automatically downloaded in the train.py or eval.py scripts.  You can turn off this behavior by setting `--download_dataset=False`. 

To see all options of `train.py`, run the `--help` or `-h` command.

```
python3 train.py --help
```

## Example Commands

- ```
  python3 train.py --batch_size=32
  ```

- ```
  python3 train.py --model_name=mtc3_model.h5 --ciphers=mtc3
  ```

- ```
  python3 train.py --model_name=custom_model_200k.h5 --ciphers=vigenere,hill --max_iter=200000 
  ```

- ```
  python3 train.py --min_train_len=100 --max_train_len=100 --min_test_len=100 --max_test_len=100
  ```

## Multi-GPU Support

CANN now supports multiple GPUs seamlessly during training:

- Before running any of the scripts, run: `export CUDA_VISIBLE_DEVICES=[gpus]`

  - Where you should replace [gpus] with a comma separated list of the index of each GPU you want to use (e.g., 0,1,2,3).
  - You should still do this if only using 1 GPU.
  - You can check the indices of your GPUs with `nvidia-smi`.

- Then, simply set the batch size to `8*num_gpus`

  with the training commands above. The training script will automatically scale the hyperparameters to the right values.

  - If you have memory to spare you can increase the batch size further, but keep it a multiple of the number of GPUs you're using.
