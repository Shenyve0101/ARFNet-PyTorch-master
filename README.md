# ARFNet-PyTorch-master

Adaptive Receptive Field Network (ARFNet) is proposed to detection various knives in natural images. 



## Installation

Install PyTorch >= 1.2.0 by selecting your environment on the website and running the appropriate command.



## Training

1.Download the pytorch official pretrained weights (retinanet_resnet50.pth), and put it in the `./model_data` dir;<br>

2.Create VOCdevkit folder and put VOC format dataset in the `./VOCdevkit` dir'. <br>

3.Generate the corresponding `txt` files with `voc_annotation.py` file before training;<br>

4.Run `train.py`.

