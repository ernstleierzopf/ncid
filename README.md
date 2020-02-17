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

- TODO: Set up the environment using one of the following methods:
   - Using [Anaconda](https://www.anaconda.com/distribution/)
     
     - Run `conda env create -f environment.yml`
   - Manually with pip
     - Set up a Python3 environment (e.g., using virtenv).
     - Install [Pytorch](http://pytorch.org/) 1.0.1 (or higher) and TorchVision.
     - Install some other packages:
       ```Shell
       # Cython needs to be installed before pycocotools
       pip install cython
       pip install opencv-python pillow pycocotools matplotlib 
       ```
   
- Before running any bash script you have to add permissions for execution by running the following command:

   ```shell
   chmod +x -R data/scripts/downloads
   ```

- If you would like to train CANN, download the basic mtc3_cipher_id dataset or the extended dataset by using `-e` or `--extended_download`. Note that this script will take a while and dump xxxxgb or respectively xxxxgb of files into `./data/mtc3_cipher_id`.

   ```bash
   sh data/scripts/downloads/mtc3_cipher_id.sh
   ```

- If you'd like to evaluate or showcase CANN on `test-dev`, download `test-dev` with this script.

  ```bash
  sh data/scripts/downloads/mtc3_cipher_id_test.sh
  ```

## Generate own data

### Generate Plaintexts

If you'd like to create your own plaintexts, you can use the `generatePlainTextFiles.py` script.  For further description read the help by using the `--help` parameter. Example usage:

```
python3 generatePlainTextFiles.py --input=../../gutenberg_en --output=../../plaintexts/gutenberg_en_plaintexts.txt split_size=10
```

### Generate Ciphertexts

If you'd like to create your own ciphertexts, you can download plaintexts extracted from the Gutenberg project and made available by us or create own plaintexts like described above. Every plaintext is seperated by the new line seperator of the operating system, so be sure to save it in the right format. Note that this script will take a while and dump xxxxgb of files into `./data/plaintexts`.

```
sh data/scripts/plaintexts.sh
```

After the plaintexts were downloaded or generated, run the `generateCipherTextFiles.py` script to generate ciphertexts. Example usage:

``` 
python3 generateCipherTextFiles.py --input=../../plaintexts/plaintexts.txt --output=../../mtc3_cipher_id/
```

There are many more parameters, which you can read about by using `--help`.

# TODO: Evaluation

Here are our CANN models (released on April 5th, 2019) along with their FPS on a Titan Xp and mAP on `test-dev`:

To evalute the model, put the corresponding weights file in the `./weights` directory and run one of the following commands. The name of each config is everything before the numbers in the file name (e.g., `yolact_base` for `yolact_base_54_800000.pth`).

## Quantitative Results

## Qualitative Results

## Benchmarking

## Texts

To see all options of `eval.py`, run the `--help` or `-h` command.

```
python3 eval.py --help
```

# TODO: Training

By default we train ciphers of the MysteryTwister C3 Challenge "Cipher ID".  Make sure to download the entire dataset using the commands above.

Example Commands

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
