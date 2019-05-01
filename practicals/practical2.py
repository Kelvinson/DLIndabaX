import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
from torchvision import transforms
import sys
from practical1 import fcn, transform, convert_state_dict

# FCN 8s
class fcn8s(fcn):
    def __init__(self, n_classes=21, transconv=False):
        super(fcn8s, self).__init__()
        self.transconv = transconv
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # Decoder
        self.fconv_block = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 256, 1),
        )
        # I am setting bias to False for reasons that will be clarified in Few-Shot Seg
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(256, self.n_classes, 1, bias=False),
        )
        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1, bias=False)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1, bias=False)

        if self.transconv:
            self.upscore2 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 4,
                                               stride=2, bias=False)
            self.upscore4 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 4,
                                               stride=2, bias=False)
            self.upscore8 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 16,
                                               stride=8, bias=False)

    def forward(self, x):
        # Extract Features
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        fconv = self.fconv_block(conv5)

        # Compute probability maps
        score = self.classifier(fconv)
        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)
        scores = [score.detach(), score_pool4.detach(),
                  score_pool3.detach()]

        if not self.transconv:
            score = F.upsample(score, score_pool4.size()[2:])
            score += score_pool4
            score = F.upsample(score, score_pool3.size()[2:])
            score += score_pool3
            out = F.upsample(score, x.size()[2:])
        else:
            upscore2 = self.upscore2(score)
            score_pool4c = self.score_pool4(conv4)[:, :, 5:5+upscore2.size()[2],
                                                         5:5+upscore2.size()[3]]
            upscore_pool4 = self.upscore4(upscore2 + score_pool4c)

            score_pool3c = self.score_pool3(conv3)[:, :, 9:9+upscore_pool4.size()[2],
                                                         9:9+upscore_pool4.size()[3]]

            out = self.upscore8(score_pool3c + upscore_pool4)[:, :, 31:31+x.size()[2],
                                                                    31:31+x.size()[3]]
        return out, scores

if __name__=="__main__":
    model = fcn8s(transconv=int(sys.argv[1]))

    # Load Weights
    device = torch.device("cuda")
    if int(sys.argv[1]):
        pass
    else:
        state = convert_state_dict(torch.load('../weights/fcn8s/fcn8s_pascal_best_model.pkl')['model_state'])
        model.load_state_dict(state)
    model.eval()
    model.to(device)

    # Load Image
    img = cv2.imread('../samples/2007_000027.jpg')
    original_img = img.copy()
    img = transform(img).unsqueeze(0)
    img = img.to(device)

    # Infer on img
    outputs, hmaps = model(img)
    pred = outputs.data.max(1)[1].cpu().numpy()

    pascal_lbls = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

    # Visualize Prediction
    plt.figure(2);plt.imshow(original_img[:,:,::-1]);
    plt.figure(1);plt.imshow(pred[0]);plt.show()

