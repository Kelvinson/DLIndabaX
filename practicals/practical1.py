import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
from torchvision import transforms

def bootstrapped_cross_entropy2d(input,
                                  target,
                                  K,
                                  weight=None,
                                  size_average=True):
    batch_size = input.size()[0]
    def _bootstrap_xentropy_single(input,
                                   target,
                                   K,
                                   weight=None,
                                   size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input,
                               target,
                               weight=weight,
                               reduce=False,
                               size_average=False,
                               ignore_index=250)
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)

# FCN 8s
class fcn(nn.Module):
    def __init__(self, n_classes=21):
        super(fcn, self).__init__()
        self.n_classes = n_classes

        # Encoder: Feature Extraction VGG-16
        self.create_encoder()

        # Decoder
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(256, self.n_classes, 1),
        )

    def create_encoder(self):
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )


    def forward(self, x):
        # Extract Features
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        # Compute probability maps
        score = self.classifier(conv5)
        out = F.upsample(score, x.size()[2:])
        return out, score.detach()

def transform(img):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])])


    img = tf(img)
    return img

def convert_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


if __name__=="__main__":
    model = fcn()

    # Load Weights
    device = torch.device("cuda")
    state = convert_state_dict(torch.load('../weights/fcn/fcn32s_pascal_best_model.pkl')['model_state'])
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
    plt.figure(2);plt.imshow(original_img[:,:,::-1]);plt.show()

    crct_cls = pascal_lbls.index('person')
    plt.figure(1);plt.imshow(pred[0]);
    plt.figure(3);plt.imshow(hmaps[0, crct_cls, ...]);plt.show()

