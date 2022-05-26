import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from resnet import resnet50
from thop import profile


if __name__ == '__main__':
    model = resnet50()
    print(model)