# AdaMatch-pytorch
A PyTorch implementation of [AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation](https://arxiv.org/abs/2106.04732).

This implementation is heavily based off of [Sayak Paul](https://github.com/sayakpaul)'s excellent [keras blog post](https://keras.io/examples/vision/adamatch/).

## How to run?
You can run this code simply by doing:
```
python run.py
```

But, note that this implementation is to be used more as a starting point, and you'll have to dig a little deeper on the python code in order to change hyperparameters, transforms and the data that is being loaded.

Running the the code as implemented here achieves the following results: 
- Accuracy on source dataset = _0.9822_ 
- Accuracy on target dataset = _0.9561_ 

The training metrics (accuracy considered over the test set) and confusion matrix (on the test set) are, respectivelly:
<img src="fig_metrics.png" alt="Training metrics"/>
<img src="fig_cm.png" alt="Confusion matrix" width="200"/>

## What's the difference from the TensorFlow implementation?
There are some very important differences:
1. While the TensorFlow implementation uses RandAugment, I don't use it as my default option. This is simply due to the fact that I didn't get the best results while doing so. Instead, I managed to get a better target accuracy by doing some DIY transforms directly on the tensors, as highlighted in `data.py`.
2. I use a ResNet18 as my classifier. It shouldn't be too hard to use other networks though, it's juts a matter of importing another network in its place.
3. The TensorFlow implementation uses `CosineDecay` for the learning rate scheduler. I tried replicating it in PyTorch with `CosineAnnealingWarmRestarts`, but I'm not sure if this is the way to do to it, so I settled with using a `StepLR` scheduler.

## Comments
I'm new in the area of unsupervised domain adaptation so I might've gotten some things wrong. If you notice anything that looks out of the ordinary feel free to open an issue. Also suggestions are very much appreciated! :)