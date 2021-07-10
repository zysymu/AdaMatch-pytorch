from torch import nn
from torchvision.models import resnet18

class Network(nn.Module):
    def __init__(self, pretrained=False, input_dim=1, n_classes=10):
        super(Network, self).__init__()

        self.resnet = resnet18(pretrained=pretrained)

        # necessary in order to use images with only 1 chnnel (MNIST and USPS data)
        self.resnet.conv1 = nn.Conv2d(input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x