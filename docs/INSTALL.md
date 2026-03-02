# Installation

**NOTE**: We have made many modifications to the original Dassl runtime. Therefore, please use the code in this repository to initialize Dassl.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n fvgpt python=3.8

# Activate the environment
conda activate fvgpt

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

* Clone FVG-PT code repository and install requirements
```bash
# Clone FVG-PT code base
git clone https://github.com/JREion/FVG-PT.git

# Install requirements (NumPy should be below 2.0.0)
cd FVG-PT/
pip install -r requirements.txt
pip install numpy==1.24.1 pip install plotnine==0.13.6

```

* Install Dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
# original source: https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop

cd ..
```
