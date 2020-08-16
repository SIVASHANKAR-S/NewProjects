import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import json
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import os
from os import listdir
import matplotlib
from matplotlib import interactive
interactive(True)
matplotlib.use('Agg')
from collections import OrderedDict

# Directories

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# criteria - Data Normalization

# Normalization of Mean and standard deviations for th respective data transforms
data_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
# Resizing the image with the following specification
data_modified = transforms.Compose([transforms.Resize(225),
                                   transforms.CenterCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

# criteria - data loading
image_datasets = datasets.ImageFolder(data_dir,transform = data_modified)
# Loading Training dataset
train_datasets = datasets.ImageFolder(train_dir,transform = data_modified)
# Loading Test dataset
test_datasets = datasets.ImageFolder(test_dir,transform = data_modified)
# Loading validation dataset
valid_datasets = datasets.ImageFolder(valid_dir,transform = data_modified)





