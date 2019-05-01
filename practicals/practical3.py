import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import deeplab_resnet
from collections import OrderedDict
import os
from os import walk
import matplotlib.pyplot as plt
import torch.nn as nn

from docopt import docopt

docstr = """Evaluate ResNet-DeepLab trained on scenes (VOC 2012),a total of 21 labels including background

Usage:
    evalpyt.py [options]

Options:
    -h, --help                  Print this message
    --visualize                 view outputs of each sketch
    --snapPrefix=<str>          Snapshot [default: VOC12_scenes_]
    --testGTpath=<str>          Ground truth path prefix [default: data/gt/]
    --testIMpath=<str>          Sketch images path prefix [default: data/img/]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 21]
    --gpu0=<int>                GPU number [default: 0]
"""

args = docopt(docstr, version='v0.1')

gpu0 = int(args['--gpu0'])
im_path = args['--testIMpath']
model = deeplab_resnet.Res_Deeplab(int(args['--NoLabels']))
model.eval()
counter = 0
model.cuda(gpu0)
snapPrefix = args['--snapPrefix']

saved_state_dict = torch.load(snapPrefix)
counter+=1
model.load_state_dict(saved_state_dict)

img = np.zeros((513,513,3));

img_temp = cv2.imread(im_path).astype(float)
img_original = img_temp
img_temp[:,:,0] = img_temp[:,:,0] - 104.008
img_temp[:,:,1] = img_temp[:,:,1] - 116.669
img_temp[:,:,2] = img_temp[:,:,2] - 122.675
img[:img_temp.shape[0],:img_temp.shape[1],:] = img_temp
output = model(Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True).cuda(gpu0))
interp = nn.UpsamplingBilinear2d(size=(513, 513))
output = interp(output[3]).cpu().data[0].numpy()
output = output[:,:img_temp.shape[0],:img_temp.shape[1]]

output = output.transpose(1,2,0)
output = np.argmax(output,axis = 2)
if args['--visualize']:
    plt.subplot(3, 1, 1)
    plt.imshow(img_original)
    plt.subplot(3, 1, 3)
    plt.imshow(output)
    plt.savefig('test.png')
