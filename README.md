# Path-BERT

This project is under active development.

## 1. Directories

* <code>[./data](./data)</code>: Contains all data files.
* <code>[./hpc_scripts](./hpc_scripts)</code>: Contains scripts for running on the HPC.
* <code>[./src](./src)</code>: All source code can be found here.

## 2. Getting Started

This project has been verified to be working correctly under the following environments:
* Python 3.8.5
* Ubuntu 20.04.2 LTS / Ubuntu 16.04.7 LTS
* CUDA 10.2 / CUDA 11.2

### 2a. Clone this repository to your local machine.

```
mkdir BERTwithKG
git clone https://github.com/IBPA/BERTwithKG.git ./BERTwithKG
```

### 2b. Create and activate virtual environment.

The following command will create and activate a virtual environment.

```
cd ./BERTwithKG
python3 -m venv env
source env/bin/activate
```

Don't forget to deactivate the virtual environment when you're done.
```
cd ./BERTwithKG
deactivate
```

### 2c. Install the required packages.

Make sure you're still in the virtual environment.

We need to install the nightly build of PyTorch (for now) due to a bug in the stable build. It seems installing packages through the ```requirements.txt``` file does not support the ```--pre``` option. Thus, install the nightly build of PyTorch as follows depending on your version of CUDA:
```
# For CUDA 11.1
pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html

# For CUDA 10.2
pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
```

Install all other required python packages.
```
pip3 install -r requirements.txt
```

### 2d. (Optional) Install DeepSpeed.

If you want to see an improvement in training time (especially when using multiple GPUs), consider using [DeepSpeed](https://www.deepspeed.ai/). If you're lucky, installing via pypi should be all that's needed.
```
pip3 install deepspeed
```

However, you may have to pre-install CPUAdam op specifically by setting the ```DS_BUILD_CPU_ADAM``` environment variable to 1. More info can be found [here](https://www.deepspeed.ai/tutorials/advanced-install/). For more information, please also refer to the huggingface's documentation to DeepSpeed [here](https://huggingface.co/transformers/master/main_classes/deepspeed.html).
```
DS_BUILD_CPU_ADAM=1 pip3 install deepspeed
```

### 2e. Run code.

1. For instructions on pre-processing, please refer to its own <code>[README](./src/preprocess/README.md)</code> file.
2. For instructions on pre-training, please refer to its own <code>[README](./src/pretrain/README.md)</code> file.
3. For instructions on fine-tuning, please refer to its own <code>[README](./src/pretrain/README.md)</code> file.

## 3. Authors

* **Jason Youn** @[https://github.com/jasonyoun](https://github.com/jasonyoun)

## 4. Contact

For any questions, please contact us at tagkopouloslab@ucdavis.edu.

## 5. Citation

We will update this section once citation information is available.

## 6. License

This project is licensed under the **Apache-2.0 License**. Please see the <code>[LICENSE](./LICENSE)</code> file for details.

## 7. Acknowledgments

* Acknowledgements go here.
* If there are people beta tested the code, help with its writing, etc. add them here.
