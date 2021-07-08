from torch import nn
from torchvision.models import resnet18

class Network(nn.Module):
    def __init__(self, pretrained=False, input_dim=1, n_classes=10):
        super(Network, self).__init__()

        default_resnet = resnet18(pretrained=pretrained)

        # necessary in order to use images with only 1 chnnel (MNIST and USPS data)
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet = nn.Sequential(
            default_resnet.bn1,
            default_resnet.relu,
            default_resnet.maxpool,
            default_resnet.layer1,
            default_resnet.layer2,
            default_resnet.layer3,
            default_resnet.layer4,
            default_resnet.avgpool
        )

        self.fc = nn.Linear(default_resnet.fc.in_features, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x