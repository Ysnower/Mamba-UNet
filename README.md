UltraLight VM-UNet

## *1.NOTE: The unofficial code for "UltraLight VM-UNet.*

Support single gpu training

## *2.prepare your own dataset*

1.The file format reference is as follows. (The image is a 24-bit jpg
image. The mask is an 8-bit png image. (0 pixel dots for background, 255
pixel dots for target))

./datasets
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
