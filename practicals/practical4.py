import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
from torchvision import transforms
import sys
from practical2 import fcn8s
from practical1 import transform, convert_state_dict

# FCN 8s
class twostreamfcn8s(fcn8s):
    def __init__(self, n_classes=21):
        super(twostreamfcn8s, self).__init__(n_classes=n_classes)
        self.mot_encoder = self.create_encoder()
        self.app_encoder = self.create_encoder()
        self.upscore = nn.Conv2d(self.n_classes, self.n_classes, 3, padding=1)

    def create_encoder(self):
        conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        encoder = nn.Sequential(conv_block1,
                                conv_block2,
                                conv_block3,
                                conv_block4,
                                conv_block5)
        return encoder

    def forward(self, x, flo):
        # Extract Features
        mot_feats = self.mot_encoder(flo)
        app_feats = self.app_encoder(x)
        feats = mot_feats * app_feats

        # Compute probability maps
        fconv = self.fconv_block(feats)
        score = self.classifier(fconv)

        out = F.upsample(score, x.size()[2:])
        out = self.upscore(out)
        return out

if __name__=="__main__":
    model = twostreamfcn8s(n_classes=2)

    # Load Weights
    device = torch.device("cuda")
#    state = convert_state_dict(torch.load('../weights/fcn8s/fcn8s_pascal_best_model.pkl')['model_state'])
#    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # Load Image
    img = cv2.imread('../samples/blackswan/00000.jpg')
    flow = cv2.imread('../samples/blackswan_of/00000.jpg')

    original_img = img.copy()
    original_flow = flow.copy()

    img = transform(img).unsqueeze(0)
    img = img.to(device)

    flow = transform(flow).unsqueeze(0)
    flow = flow.to(device)


    # Infer on img
    outputs, hmaps = model(img, flow)
    pred = outputs.data.max(1)[1].cpu().numpy()

    # Visualize Prediction
    plt.figure(3);plt.imshow(original_img[:,:,::-1]);
    plt.figure(2);plt.imshow(original_flow[:,:,::-1]);
    plt.figure(1);plt.imshow(pred[0]);plt.show()

