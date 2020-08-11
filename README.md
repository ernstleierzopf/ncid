# cann

A neuronal network to detect and analyse ciphers from historical texts.

# CANN - Crypto Analysis with Neural Networks

This project contains code for the detection and classification of ciphers to classical algorithms by using a neuronal network. In Future other parts of the crypto analysis will be implemented.

# Installation

- Clone this repository and enter it:
  ```Shell
  git clone https://github.com/dITySoftware/cann
  cd cann
  ```

- Set up the environment with the following method:
   - Set up a Python3.7 or higher environment (e.g., using virtenv (see https://gist.github.com/Geoyi/d9fab4f609e9f75941946be45000632b)).
   
   - Install all needed packages:
     
     ```Shell
     pip3 install tensorflow tensorflow_datasets numpy scikit_learn
     ```

- Install the recommended and tested versions by using requirements.txt:

  ```
  pip3 install -r requirements.txt
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

# Evaluation

Here is our CANN model (released on April 22nd, 2020): [M24.h5](https://drive.google.com/open?id=1VZQ1eiJSV9Z3mfDIhvRTWHmaMcYjSRZb)

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

# Unit-Tests

Multiple Unit-Tests ensure the functionality of the implemented ciphers and the TextLine2CipherStatisticsDataset. 

Every test case can be executed by using following command in the main directory:

```
python3 -m unittest discover -s unit -p '*Test.py'
```

Single test classes can be executed with this command:

  ```
python3 -m unittest <path/to/test/class>
  ```

for example:

  ```
python3 -m unittest unit/cipherTypeDetection/textLine2CipherStatisticsDataset.py
  ```

# Quantitative Results

Following are our training results from a DGX-1 with 2 GPUs. Model M0 is trained with a Logistic Regression Model to set a simple baseline. Models M1-M20 are trained with a 5 Hidden Layers Model. From Model M20 and further on other filter functions, where no padding is added (playfair, hill), were used. M1 through M23 were trained and tested with wrong filtering of the playfair data. Instead of adding an x between double characters the second character was replaced with it. M24 trained and tested with the right filter function.

| Model                           | M0           | M1                          | M2                          | M3                          | M4                          | M5                          | M6                          | M7                          | M8                          | M9                          | M10                         | M11                         |
| ------------------------------- | ------------ | --------------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- |
| Total accuracy                  | 0.675771     | 0.863456                    | 0.874496                    | 0.863978                    | 0.864356                    | 0.860269                    | 0.867630                    | 0.854944                    | 0.864138                    | 0.855068                    | 0.863595                    | 0.869381                    |
| Columnar Transposition accuracy | 1.0          | 0.998963                    | 0.998537                    | 0.998824                    | 0.996024                    | 0.998976                    | 0.998226                    | 0.998108                    | 0.999247                    | 0.999578                    | 0.999311                    | 0.999135                    |
| Hill accuracy                   | 0.456083     | 0.740203                    | 0.816435                    | 0.856772                    | 0.755070                    | 0.681300                    | 0.672694                    | 0.766543                    | 0.749240                    | 0.769820                    | 0.796152                    | 0.717530                    |
| Playfair accuracy               | 0.934648     | 0.977118                    | 0.970606                    | 0.970552                    | 0.981339                    | 0.974251                    | 0.989121                    | 0.950802                    | 0.972833                    | 0.981176                    | 0.981056                    | 0.989141                    |
| Simple Substitution accuracy    | 0.631038     | 0.981564                    | 0.989968                    | 0.981133                    | 0.976279                    | 0.986623                    | 0.989388                    | 0.986896                    | 0.980652                    | 0.978494                    | 0.978950                    | 0.982225                    |
| Vigenere accuracy               | 0.357090     | 0.619435                    | 0.596935                    | 0.512612                    | 0.613072                    | 0.660195                    | 0.688722                    | 0.572376                    | 0.618722                    | 0.546277                    | 0.562507                    | 0.658876                    |
| Model Architecture              | Log. Regress | 5-Hidden-Layer Crossentropy | 5-Hidden-Layer Crossentropy | 5-Hidden-Layer Crossentropy | 5-Hidden-Layer Crossentropy | 5-Hidden-Layer Crossentropy | 5-Hidden-Layer Crossentropy | 5-Hidden-Layer Crossentropy | 5-Hidden-Layer Crossentropy | 5-Hidden-Layer Crossentropy | 5-Hidden-Layer Crossentropy | 5-Hidden-Layer Crossentropy |
| batch_size                      | 16           | 16                          | 16                          | 16                          | 16                          | 16                          | 16                          | 16                          | 16                          | 16                          | 16                          | 16                          |
| iterations (in Mio)             | 10           | 10                          | 50                          | 10                          | 10                          | 10                          | 10                          | 10                          | 10                          | 10                          | 10                          | 10                          |
| epochs                          | 1            | 1                           | 1                           | 3                           | 1                           | 1                           | 1                           | 1                           | 1                           | 1                           | 1                           | 1                           |
| keep_unknown_symbols            | False        | False                       | False                       | False                       | False                       | False                       | False                       | False                       | False                       | False                       | False                       | False                       |
| train text length               | 100 \| 100   | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  |
| test text length                | 100 \| 100   | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  | 100 \| 100                  |
| key lengths                     | 4 - 16       | 4 - 16                      | 4 - 16                      | 4 - 16                      | 4 - 16                      | 4 - 16                      | 4 - 16                      | 4 - 16                      | 4 - 16                      | 4 - 16                      | 4 - 16                      | 4 - 16                      |
| dataset size                    | 65000        | 65000                       | 65000                       | 65000                       | 10010                       | 65000                       | 65000                       | 65000                       | 65000                       | 65000                       | 65000                       | 65000                       |
| workers                         | 50           | 50                          | 50                          | 50                          | 50                          | 50                          | 50                          | 50                          | 50                          | 50                          | 50                          | 50                          |
| training time                   | 25m 36s      | 36m 37s                     | 2h 41m 28s                  | 1h 23m 51s                  | 11h 59m 49s                 | 39m 27s                     | 39m 11s                     | 39m 11s                     | 59m 3s                      | 39m 52s                     | 40m 26s                     | 1h 1m 20s                   |
| Unigrams                        | ✔            | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           |
| Bigrams                         | ✔            | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           |
| Ny-Bigrams (range intervals)    | **X**        | **X**                       | **X**                       | **X**                       | ✔ 2-7                       | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       |
| Index of Coincidence Unigrams   | ✔            | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           |
| Index of Coincidence Bigrams    | ✔            | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           | ✔                           |
| Has Letter J Check              | **X**        | **X**                       | **X**                       | **X**                       | **X**                       | ✔                           | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | ✔                           |
| Has Doubles Check               | **X**        | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | ✔                           | **X**                       | **X**                       | **X**                       | **X**                       | ✔                           |
| Chi Square                      | **X**        | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | ✔                           | **X**                       | **X**                       | **X**                       | **X**                       |
| Pattern Repititions Count       | **X**        | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | ✔                           | **X**                       | **X**                       | ✔                           |
| Shannon's Entropy               | **X**        | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | ✔                           | **X**                       | **X**                       |
| Autocorrelation Average         | **X**        | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | **X**                       | ✔                           | ✔                           |