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

def compute_weight(embeddings, nclasses, labels, original_weight, alpha):
    imp_weight = embeddings.mean(0).squeeze()

    # Add imprinted weights for -ve samples that occurred in support image
    for c in range(nclasses):
        if len(labels[labels==c]) != 0:
            temp = original_weight[c, ...].squeeze()
            temp = (1-alpha)*temp + alpha*imp_weight[c].cuda()
            temp = temp / temp.norm(p=2)
            original_weight[c, ...] = temp.unsqueeze(1).unsqueeze(1)

    # Add imprinted weights for + sample (last class)
    imp_weight[-1] = imp_weight[-1] / imp_weight[-1].norm(p=2)
    imp_weight = imp_weight[-1].unsqueeze(0).unsqueeze(2).unsqueeze(3)
    weight = torch.cat((original_weight, imp_weight.cuda()), 0)
    return weight


def masked_embeddings(fmap_shape, label, fconv_norm, n_classes):
    label = label.unsqueeze(0).unsqueeze(0)
    fconv_norm = nn.functional.interpolate(fconv_norm,
                                      size=(int(label.shape[2]), int(label.shape[3])),
                                      mode='nearest')
    fconv_pooled = torch.zeros(fmap_shape[0], n_classes+1, fmap_shape[1], 1, 1)
    for i in range(int(fconv_norm.shape[1])):
        temp = fconv_norm[:, i, ...]
        for c in range(n_classes+1):
            if len(temp[label[0]==c]) == 0:
                tempv = 0
            else:
                tempv = temp[label[0]==c].mean()
            fconv_pooled[:, c, i, 0, 0] = tempv
    return fconv_pooled

# FCN 8s
class imprint_fcn8s(fcn):
    def __init__(self, n_classes=21):
        super(imprint_fcn8s, self).__init__(n_classes=n_classes)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=4, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=4, padding=4),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.fconv_block = nn.Sequential(nn.Conv2d(512, 256, 1))
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(256, self.n_classes, 1, bias=False),
        )
        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1, bias=False)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1, bias=False)


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

        score = F.upsample(score, score_pool4.size()[2:])
        score += score_pool4
        score = F.upsample(score, score_pool3.size()[2:])
        score += score_pool3
        out = F.upsample(score, x.size()[2:])

        return out

    def extract(self, x, label):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        fconv = self.fconv_block(conv5)

        fconv_pooled = masked_embeddings(fconv.shape, label, fconv,
                                         self.n_classes)
        conv3_pooled = masked_embeddings(conv3.shape, label, conv3,
                                         self.n_classes)
        conv4_pooled = masked_embeddings(conv4.shape, label, conv4,
                                         self.n_classes)

        return fconv_pooled, conv4_pooled, conv3_pooled

    def imprint(self, images, labels, alpha):
        with torch.no_grad():
            embeddings = None
            for ii, ll in zip(images, labels):
                #ii = ii.unsqueeze(0)
                ll = ll[0]
                if embeddings is None:
                    embeddings, early_embeddings, vearly_embeddings = self.extract(ii, ll)
                else:
                    embeddings_, early_embeddings_, vearly_embeddings_ = self.extract(ii, ll)
                    embeddings = torch.cat((embeddings, embeddings_), 0)
                    early_embeddings = torch.cat((early_embeddings, early_embeddings_), 0)
                    vearly_embeddings = torch.cat((vearly_embeddings, vearly_embeddings_), 0)

            # Imprint weights for last score layer
            nclasses = self.n_classes
            self.n_classes = 17
            nchannels = embeddings.shape[2]

            weight = compute_weight(embeddings, nclasses, labels,
                                         self.classifier[2].weight.data, alpha=alpha)
            self.classifier[2] = nn.Conv2d(nchannels, self.n_classes, 1, bias=False)
            self.classifier[2].weight.data = weight

            weight4 = compute_weight(early_embeddings, nclasses, labels,
                                     self.score_pool4.weight.data, alpha=alpha)
            self.score_pool4 = nn.Conv2d(512, self.n_classes, 1, bias=False)
            self.score_pool4.weight.data = weight4

            weight3 = compute_weight(vearly_embeddings, nclasses, labels,
                                     self.score_pool3.weight.data, alpha=alpha)
            self.score_pool3 = nn.Conv2d(256, self.n_classes, 1, bias=False)
            self.score_pool3.weight.data = weight3

            assert self.classifier[2].weight.is_cuda
            assert self.score_pool3.weight.is_cuda
            assert self.score_pool4.weight.is_cuda
            assert self.score_pool3.weight.data.shape[1] == 256
            assert self.classifier[2].weight.data.shape[1] == 256
            assert self.score_pool4.weight.data.shape[1] == 512

if __name__=="__main__":
    model = imprint_fcn8s(n_classes=16)

    # Load Weights
    device = torch.device("cuda")
    state = convert_state_dict(torch.load('../weights/dilated_fcn8s_fold0/dilated_fcn8s_pascal_best_model.pkl')['model_state'])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # Load Image
    sprt_img = cv2.imread('../samples/sprt_img.png')
    sprt_lbl = cv2.imread('../samples/sprt_gt.png', 0)

    img = cv2.imread('../samples/qry_img.png')
    original_img = img.copy()
    original_sprt = sprt_img.copy()
    original_gt = sprt_lbl.copy()

    img = transform(img).unsqueeze(0)
    img = img.to(device)

    sprt_img = transform(sprt_img).unsqueeze(0)
    sprt_img = sprt_img.to(device)

    sprt_lbl = torch.from_numpy(np.array(sprt_lbl)).long()
    sprt_lbl[sprt_lbl == 255] = 0
    sprt_lbl = sprt_lbl.to(device).unsqueeze(0)

    # Infer on img
    model.imprint([sprt_img], [sprt_lbl], alpha=0.25821)
    outputs = model(img)
    pred = outputs.data.max(1)[1].cpu().numpy()

    pascal_lbls = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

    # Visualize Prediction
    plt.figure(2);plt.imshow(original_img[:,:,::-1]);
    plt.figure(3);plt.imshow(original_sprt[:,:,::-1]);
    plt.figure(4);plt.imshow(original_gt)
    plt.figure(1);plt.imshow(pred[0]);plt.show()

