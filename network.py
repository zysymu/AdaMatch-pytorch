from torch import nn
from torchvision.models import resnet18

class Encoder(nn.Module):
    def __init__(self, features_size=256):
        """
        ResNet based neural network that receives images and encodes them into an array of size `features_size`.

        Arguments:
        ----------features_size: int
            Size of encoded features array.
        """

        super(Encoder, self).__init__()
        
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, features_size)

    def forward(self, x):
        x = self.resnet(x)
        return x

class Classifier(nn.Module):
    def __init__(self, features_size=256, n_classes=10):
        """
        Neural network that receives an array of size `features_size` and classifies it into `n_classes` classes.

        Arguments:
        ----------
        features_size: int
            Size of encoded features array.

        n_classes: int
            Number of classes to classify the encoded array into.
        """

        super(Classifier, self).__init__()
        self.fc = nn.Linear(features_size, n_classes)

    def forward(self, x):
        x = self.fc(x)
        return x