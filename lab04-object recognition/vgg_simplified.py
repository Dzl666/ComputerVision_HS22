import torch
import torch.nn as nn
import math

class Vgg(nn.Module):
    def __init__(self, fc_layer=512, classes=10):
        super(Vgg, self).__init__()
        """ Initialize VGG simplified Module
        Args: 
            fc_layer: input feature number for the last fully MLP block
            classes: number of image classes
        """
        self.fc_layer = fc_layer
        self.classes = classes

        conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        conv5 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.ConvBlock1 = nn.Sequential(
            conv1, nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.ConvBlock2 = nn.Sequential(
            conv2, nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.ConvBlock3 = nn.Sequential(
            conv3, nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.ConvBlock4 = nn.Sequential(
            conv4, nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.ConvBlock5 = nn.Sequential(
            conv5, nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, self.fc_layer, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.fc_layer, self.classes, bias=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        """
        :param x: input image batch tensor, [bs, 3, 32, 32]
        :return: score: predicted score for each class (10 classes in total), [bs, 10]
        """
        score = None
        
        output = self.ConvBlock1(x)
        output = self.ConvBlock2(output)
        output = self.ConvBlock3(output)
        output = self.ConvBlock4(output)
        output = self.ConvBlock5(output) # [bs, 512, 1, 1]
        output = torch.flatten(output, 1, 3) # [bs, 512]
        score = self.classifier(output) # [bs, 10]

        return score

