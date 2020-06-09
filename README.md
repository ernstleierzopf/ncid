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

# Qualitative Results

Following are our training results from a DGX-1 with 2 GPUs. Model M0 is trained with a Logistic Regression Model to set a simple baseline. Models M1-M20 are trained with a 5 Hidden Layers Model. From Model M20 and further on other filter functions, where no padding is added (playfair, hill), were used. M1 through M23 were trained and tested with wrong filtering of the playfair data. Instead of adding an x between double characters the second character was replaced with it. M24 trained and tested with the right filter function.

| Model                           | M0         | M1         | M2         | M3         | M4               | M5          | M6         | M7         | M8         | M9         | M10        | M11        | M12        | M13        | M14        | M15        | M16              | M17              | M18         | M19        | M20                                  | M21           | M22                     | M23                     | M24                                  |
| ------------------------------- | ---------- | ---------- | ---------- | ---------- | ---------------- | ----------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------------- | ---------------- | ----------- | ---------- | ------------------------------------ | ------------- | ----------------------- | ----------------------- | ------------------------------------ |
| Total accuracy                  |            | 0.868428   | 0.879094   | 0.873371   | 0.866920         | 0.869525    | 0.866719   | 0.872321   | 0.857106   | 0.865288   | 0.860668   | 0.869005   | 0.870589   | 0.831755   | 0.932003   | 0.841680   | 0.882806         | 0.877725         | 0.862423    | 0.868907   | 0.905355                             | 0.676492      | 0.790592                | 0.790473                | 0.907627                             |
| Columnar Transposition accuracy |            | 0.999642   | 0.999688   | 0.999183   | 0.999377         | 0.999146    | 0.999997   | 0.999750   | 0.998435   | 0.999697   | 0.999626   | 0.999967   | 0.999938   | 0.999987   | 1.0        | 0.991232   | 0.999582         | 0.999983         | 0.998621    | 0.999543   | 0.998701                             |               |                         |                         | 0.998897                             |
| Hill accuracy                   |            | 0.744559   | 0.777486   | 0.760050   | 0.718811         | 0.712228    | 0.698685   | 0.818429   | 0.800454   | 0.681740   | 0.789729   | 0.714444   | 0.637963   | 0.581396   | 0.893760   | 0.618737   | 0.729221         | 0.776212         | 0.771862    | 0.816271   | 0.833647                             | 0.683305      | 0.866652                | 0.815824                | 0.874263                             |
| Playfair accuracy               |            | 0.968207   | 0.979629   | 0.973104   | 0.961294         | 0.966978    | 0.962283   | 0.990377   | 0.965593   | 0.954886   | 0.988185   | 0.971831   | 0.987463   | 0.968792   | 0.992526   | 0.963328   | 0.995735         | 0.983058         | 0.956979    | 0.976642   | 0.959890                             |               |                         |                         | 0.966945                             |
| Simple Substitution accuracy    |            | 0.986788   | 0.981202   | 0.980970   | 0.990485         | 0.981703    | 0.985793   | 0.988000   | 0.967649   | 0.989141   | 0.969072   | 0.982763   | 0.990293   | 0.962344   | 0.976734   | 0.988698   | 0.984766         | 0.982767         | 0.985382    | 0.981268   | 0.992638                             |               |                         |                         | 0.995404                             |
| Vigenere accuracy               |            | 0.642945   | 0.657463   | 0.653549   | 0.664633         | 0.687572    | 0.686837   | 0.565047   | 0.553396   | 0.700978   | 0.556728   | 0.676018   | 0.737288   | 0.646253   | 0.796994   | 0.646402   | 0.704723         | 0.646606         | 0.599269    | 0.570812   | 0.741900                             | 0.669679      | 0.714533                | 0.765122                | 0.702627                             |
| batch_size                      | 16         | 16         | 16         | 16         | 16               | 16          | 16         | 16         | 16         | 16         | 16         | 16         | 16         | 16         | 16         | 16         | 16               | 16               | 16          | 16         | 16                                   | 16            | 16                      | 16                      | 16                                   |
| iterations (in Mio)             | 10         | 10         | 50         | 10         | 10               | 10          | 10         | 10         | 10         | 10         | 10         | 10         | 10         | 0,77       | 0,77       | 10         | 50               | 50               | 10          | 10         | 50                                   | 10            | 10                      | 10                      | 50                                   |
|                                 | 1          | 1          | 1          | 3          | 1                | 1           | 1          | 1          | 1          | 1          | 1          | 1          | 1          | 1          | 1          | 1          | 1                | 1                | 1           | 1          | 1                                    | 1             | 1                       | 1                       | 1                                    |
| keep_unknown_symbols            | False      | False      | False      | False      | False            | False       | False      | False      | False      | False      | False      | False      | False      | False      | False      | False      | False            | False            | False       | False      | False                                | False         | False                   | False                   | False                                |
| train text length               | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100       | 100 \| 100  | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100       | 100 \| 100       | 100 \| 100  | 100 \| 100 | 100 \| 100                           | 100 \| 100    | 100 \| 100              | 100 \| 100              | 100 \| 100                           |
| test text length                | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100       | 100 \| 100  | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100 | 100 \| 100       | 100 \| 100       | 100 \| 100  | 100 \| 100 | 100 \| 100                           | 100 \| 100    | 100 \| 100              | 100 \| 100              | 100 \| 100                           |
| key lengths                     | 4 - 16     | 4 - 16     | 4 - 16     | 4 - 16     | 4 - 16           | 4 - 16      | 4 - 16     | 4 - 16     | 4 - 16     | 4 - 16     | 4 - 16     | 4 - 16     | 4 - 16     | 13         | 4          | 13         | 4 - 16           | 4 - 16           | 4-16        | 4-16       | Col,Vig:5,10,15,20 Playfair: 6,7,8,9 | Hill,Vigenere | Hill,Vigenere           | Hill,Vigenere           | Col,Vig:5,10,15,20 Playfair: 6,7,8,9 |
| dataset size                    | 65000      | 65000      | 65000      | 65000      | 65000            | 10010       | 65000      | 65000      | 65000      | 65000      | 65000      | 65000      | 65000      | 15400      | 15400      | 65000      | 65000            | 65000            | 65000       | 65000      | 65000                                | 65000         | 65000                   | 100000                  | 65000                                |
| workers                         | 50         | 50         | 50         | 50         | 50               | 50          | 50         | 50         | 50         | 50         | 50         | 50         | 50         | 50         | 50         | 50         | 50               | 50               | 50          | 50         | 20                                   | 50            | 20                      | 8                       | 20                                   |
| training time                   |            | 1h 19m 5s  | 6h 45m 49s | 2h 47m 41s | 2h 32m 32s       | 15h 21m 46s | 1h 31m 44s | 1h 20m 37s | 1h 17m 5s  | 1h 53m 21s | 1h 15m 52s | 1h 17m 41s | 1h 46m 53s | 8m 22s     | 8m 34s     | 1h 28m 48s | 11h 43m 56s      | 11h 10m 0s       | 1h  18m 10s | 1h 14m 38s | 16h 35m 1s                           | 1h 29m 14s    | 3h 50m 30s              | 4h 3m 25s               | 16h 31m 13s                          |
| Unigrams                        | ✔          | ✔          | ✔          | ✔          | ✔                | ✔           | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔                | ✔                | ✔           | ✔          | ✔                                    | ✔             | ✔                       | ✔                       | ✔                                    |
| Bigrams                         | ✔          | ✔          | ✔          | ✔          | ✔                | ✔           | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔                | ✔                | ✔           | ✔          | ✔                                    | ✔             | ✔                       | ✔                       | ✔                                    |
| Ny-Bigrams (range intervals)    | **X**      | **X**      | **X**      | **X**      | ✔ 2-15 (average) | ✔ 2-15      | **X**      | **X**      | **X**      | **X**      | **X**      | **X**      | **X**      | **X**      | **X**      | **X**      | ✔ 2-15 (average) | ✔ 2-15 (average) | **X**       | **X**      | ✔ (interval=5,10,20,25)              | **X**         | ✔ (interval=5,10,20,25) | ✔ (interval=5,10,20,25) | ✔ (interval=5,10,20,25)              |
| Index of Coincidence Unigrams   | ✔          | ✔          | ✔          | ✔          | ✔                | ✔           | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔                | ✔                | ✔           | **X**      | ✔                                    | ✔             | ✔                       | ✔                       | ✔                                    |
| Index of Coincidence Bigrams    | ✔          | ✔          | ✔          | ✔          | ✔                | ✔           | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔          | ✔                | ✔                | **X**       | ✔          | ✔                                    | ✔             | ✔                       | ✔                       | ✔                                    |
| Has Letter J Check              | **X**      | **X**      | **X**      | **X**      | **X**            | **X**       | ✔          | **X**      | **X**      | **X**      | **X**      | **X**      | ✔          | **X**      | **X**      | **X**      | **X**            | **X**            | **X**       | **X**      | **X**                                | **X**         | **X**                   | **X**                   | **X**                                |
| Has Doubles Check               | **X**      | **X**      | **X**      | **X**      | **X**            | **X**       | **X**      | ✔          | **X**      | **X**      | **X**      | **X**      | ✔          | **X**      | **X**      | **X**      | ✔                | **X**            | **X**       | **X**      | **X**                                | **X**         | **X**                   | **X**                   | **X**                                |
| Chi Square                      | **X**      | **X**      | **X**      | **X**      | **X**            | **X**       | **X**      | **X**      | ✔          | **X**      | **X**      | **X**      | **X**      | **X**      | **X**      | **X**      | **X**            | **X**            | **X**       | **X**      | **X**                                | **X**         | **X**                   | **X**                   | **X**                                |
| Pattern Repititions Count       | **X**      | **X**      | **X**      | **X**      | **X**            | **X**       | **X**      | **X**      | **X**      | ✔          | **X**      | **X**      | ✔          | **X**      | **X**      | **X**      | **X**            | **X**            | **X**       | **X**      | **X**                                | **X**         | **X**                   | **X**                   | **X**                                |
| Shannon's Entropy               | **X**      | **X**      | **X**      | **X**      | **X**            | **X**       | **X**      | **X**      | **X**      | **X**      | ✔          | **X**      | **X**      | **X**      | **X**      | **X**      | **X**            | **X**            | **X**       | **X**      | **X**                                | **X**         | **X**                   | **X**                   | **X**                                |
| Autocorrelation Average         | **X**      | **X**      | **X**      | **X**      | **X**            | **X**       | **X**      | **X**      | **X**      | **X**      | **X**      | ✔          | ✔          | **X**      | **X**      | **X**      | **X**            | **X**            | **X**       | **X**      | **X**                                | **X**         | **X**                   | **X**                   | **X**                                |
