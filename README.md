# AdaMatch-pytorch
A PyTorch implementation of [AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation](https://arxiv.org/abs/2106.04732).

This implementation is heavily based off of [Sayak Paul](https://github.com/sayakpaul)'s excellent [keras blog post](https://keras.io/examples/vision/adamatch/). I also used [Ang Yi Zhe](https://github.com/yizhe-ang)'s [PyTorch implementation](https://github.com/yizhe-ang/AdaMatch-PyTorch) to fix some problems that my initial implementation had.

## How to run?
You can run this code simply by doing:
```
python run.py
```

Note that this implementation is to be used more as a starting point, and you'll have to dig a little deeper on the python code in order to change hyperparameters, transforms and the data that is being loaded.

Running the the code on a MNIST -> USPS adapation using a _ResNet18_ architecture:
- Accuracy on source dataset = _0.9822_ 
- Accuracy on target dataset = _0.9561_ 

Some differences:
- Network used: due to limited computing resources my implementation uses a _ResNet18_ architecture. This could be easily changed by importing other models on the `network.py` file.
- The paper uses _[CTAugment](https://arxiv.org/abs/1911.09785)_ for strong augmentations. I implemented a pipeline similar to _CTAugment_ using the availabe _torchvision_ transforms.
- The original implementation uses a cosine decay for the learning rate scheduler. I tried using _PyTorch_'s cosine-related learning rate schedulers and didn't manage to achieve good results, so I used a simple step learning rate scheduler.


#### If you notice anything that looks out of the ordinary feel free to open an issue. Suggestions are very much appreciated! :)
