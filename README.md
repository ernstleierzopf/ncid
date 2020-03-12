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


## Generate Plaintexts (Optional)

If you'd like to create your own plaintexts, you can use the `generatePlainTextFiles.py` script. Therefore you first need to download some texts, for example the Gutenberg Library. You can do that by using following command, which downloads all English e-books compressed with zip.  Note that this script can take a while and dumps about 14gb of files into `./data/plaintexts`.

```shell
wget -m -H -nd "http://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en" -e use_proxy=yes -e http_proxy=46.101.1.221:80 > /tmp/wget-log 2>&1
```

The `generatePlainTextFiles.py` script automatically unpacks the zips, with the parameter `--restructure_directory`.  Every line in a plaintext is seperated by a '\n', so be sure to save it in the right format or use the `generatePlainTextFiles.py` script to reformat all files from '\r\n' to '\n'. For further description read the help by using the `--help` parameter. Example usage:

```
python3 generatePlainTextFiles.py --directory=../gutenberg_en --restructure_directory=true
```

# TODO: Evaluation

Here are our CANN models (released on April 5th, 2019) along with their evaluations per second on the DGX-1 and the results on random data:

| Model                                       | evaluations / second | cipher \| accuracy rate matrix on 100k lines of plaintexts   |
| ------------------------------------------- | -------------------- | ------------------------------------------------------------ |
| mtc3_model.h5 (Link)                        | 22.03                | simple_substitution\| accuracy rate1 <br />vigenere\| <br />columnar_transposition\| <br />playfair\| <br />hill\| |
| mtc3_logistic_regression_baseline.h5 (Link) | 24                   | simple_substitution\| accuracy rate1 <br />vigenere\| <br />columnar_transposition\| <br />playfair\| <br />hill\  |
|                                             |                      |                                                              |

To evalute the model, put the corresponding weights file in the `./weights` directory and run one of the following commands. The name of each config is everything before the numbers in the file name (e.g., `yolact_base` for `yolact_base_54_800000.pth`).

## Quantitative Results

## Qualitative Results

## Benchmarking

## Texts

To see all options of `eval.py`, run the `--help` or `-h` command.

```
python3 eval.py --help
```

# Training

By default we train ciphers of the MysteryTwister C3 Challenge "Cipher ID".  By Default the already filtered plaintexts are downloaded in the train.py script.  You can turn off this behavior by setting `--download_dataset=False`. 

To see all options of `train.py`, run the `--help` or `-h` command.

```
python3 train.py --help
```

## Example Commands

- ```
  python3 train.py --batch_size=4096
  ```

- ```
  python3 train.py --model_name=mtc3_model.h5 --ciphers=mtc3
  ```

- ```
  python3 train.py --model_name=custom_model_200k.h5 --ciphers=vigenere,hill --max_iter=200000 
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
  - If you want to allocate the images per GPU specific for different GPUs, you can use `--batch_alloc=[alloc]` where [alloc] is a comma seprated list containing the number of images on each GPU. This must sum to `batch_size`.

## TODO: Logging

CANN now logs training and validation information by default. You can disable this with `--no_log`. A guide on how to visualize these logs is coming soon, but now you can look at `LogVizualizer` in `utils/logger.py` for help.
