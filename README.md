# SEANet
Code of 'Coupled Rain Streak and Background Estimation via Separable Element-wise Attention'<br/>
YINJIE TAN, QIANG WEN, JING QIN, JIANBO JIAO, GUOQIANG HAN,SHENGFENG HE<br/>
IEEE ACCESS

## Paper
[Coupled Rain Streak and Background Estimation via Separable Element-wise Attention](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8963735)

## Requirement
- Ubuntu 16.04 
- CUDA 10.0
- Python 3.5
- Numpy
- Pytorch 1.0
- PIL
- Matplotlib
## Dataset
All rain dataset can be found here. [DerainZoo](https://github.com/nnUyi/DerainZoo)<br/><br/>
If use the default options, a directory `../rain_dataset` should be created to store datasets and each dataset should be placed
in a sub-folder. During training, the trained dataset can be modified by option `setname` (corresponding to the sub-folder name)
in `./options/base_options.py`.<br/>
<br/>
Create folders `train`, `val` and 'test' in each dataset folder. For folders `train` and `val`, input image and groundtruth image
are concatenated in width dimension.

- Example:<br/>
For 'Rain_200H' dataset, if we name its sub-folder '200h', the path for training data and validation data is 
`../rain_dataset/200h/train` and `../rain_dataset/200h/val`

## Training and Testing
- Train a model:
``` bash
python3 ./train.py 
``` 
- Test a model:
``` bash
python3 ./test.py 
``` 
More options can be modified at directory `./options`.<br/
The detail code for our model can be found at `./models/networks.py` and `seaunet_model.py`.
The training model and log of losses are stored in `./checkpoints` and the validation results (PSNR,SSIM) 
are stored in `./results`.

## Acknowledgments
The framework of this code is based on [pytorch-CycleGAN-and-pix2pix] (https://github.com/amikey/pytorch-CycleGAN-and-pix2pix).<br/>
