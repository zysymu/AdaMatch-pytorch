# AdaMatch-pytorch
A PyTorch implementation of [AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation](https://arxiv.org/abs/2106.04732).

This implementation is heavily based off of [Sayak Paul](https://github.com/sayakpaul)'s excellent [keras blog post](https://keras.io/examples/vision/adamatch/).

## How to run?
You can run this code simply by doing:
```
python run.py
```

But, note that this implementation is to be used more as a starting point, and you'll have to dig a little deeper on the python code in order to change hyperparameters, transforms and the data that is being loaded.

## What's the difference from the TensorFlow implementation?
There are some very important differences:
1. I don't use RandAugment for the strong augmentations. The main reason is because I couldn't get available RandAugment PyTorch implementations to work with my data. Also, from what I've seen, current RandAugment implementations use `PIL` for most transforms, which could end up in information loss if you're working with certain delicate images (exactly the domain that I plan to use AdaMatch on). To compensate for it I did some DIY transforms that are very hacky, but seem to get the job done.
2. I use a ResNet18 as my classifier. It shouldn't be too hard to use other networks though, it's juts a matter of importing another network in its place.
3. The TensorFlow implementation uses `CosineDecay` for the learning rate scheduler. I tried replicating it in PyTorch with `CosineAnnealingWarmRestarts`, but I'm not sure if this is the way to do to it, so I settled with using a `StepLR` scheduler.

## Comments
I'm new in the area of unsupervised domain adaptation so I might've gotten some things wrong. If you notice anything that looks out of the ordinary feel free to open an issue. Also suggestions are very much appreciated! :)