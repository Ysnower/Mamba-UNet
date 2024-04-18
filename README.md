UltraLight VM-UNet

## *1.NOTE: The unofficial code for "UltraLight VM-UNet.*

Support single gpu training
The environment</br>
```
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

## *2.prepare your own dataset*

1.The file format reference is as follows. (The image is a 24-bit jpg
image. The mask is an 8-bit png image. (0 pixel dots for background, 255
pixel dots for target))

datasets/

├── train

│   ├── images

│   └── masks

└── val

        ├── images
    
        └── masks



2.config_setting.visual_imgs=True,wirte 10 images to the folder and ensure that the label after data augmentation is correct.
Data augmentation may cause label errors, so check it!

## *3.Train the UltraLight VM-UNet.*

`python train.py`

## *4.pytorch inference*

`python inference.py`

## 5.reference

[**[UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet)**](https://github.com/wurenkai/UltraLight-VM-UNet)
