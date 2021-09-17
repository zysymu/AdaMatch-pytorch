from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import random

def _get_transforms():
    """
    The AdaMatch paper uses CTAugment as its strong augmentations. I'm going to
    create a pipeline of transforms similar to the ones used by CTAugment.
    """

    train_transform_weak = transforms.Compose([transforms.ToTensor(),
                                               transforms.Resize(28),
                                               transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3))
                                               ])

    train_transform_strong = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize(28),
                                                 transforms.RandomAutocontrast(),
                                                 transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
                                                 #transforms.RandomEqualize(), # only on PIL images
                                                 transforms.RandomInvert(),
                                                 #transforms.RandomPosterize(random.randint(1, 8)), # only on PIL images
                                                 transforms.RandomAdjustSharpness(random.uniform(0, 1)),
                                                 transforms.RandomSolarize(random.uniform(0, 1)),
                                                 transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
                                                 transforms.RandomErasing()
                                                 ])


    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(28)
                                         ])

    return train_transform_weak, train_transform_strong, test_transform

def get_dataloaders(download_path, batch_size_source=32, workers=2):
    train_transform_weak, train_transform_strong, test_transform = _get_transforms()

    BATCH_SIZE_source = batch_size_source
    BATCH_SIZE_target = 3 * BATCH_SIZE_source

    # source datasets and dataloaders
    source_dataset_train_weak = torchvision.datasets.MNIST(download_path, train=True, download=True, transform=train_transform_weak)
    source_dataset_train_strong = torchvision.datasets.MNIST(download_path, train=True, download=True, transform=train_transform_strong)
    source_dataset_test = torchvision.datasets.MNIST(download_path, train=False, download=True, transform=test_transform)

    source_dataloader_train_weak = DataLoader(source_dataset_train_weak, shuffle=False, batch_size=BATCH_SIZE_source, num_workers=workers)
    source_dataloader_train_strong = DataLoader(source_dataset_train_strong, shuffle=False, batch_size=BATCH_SIZE_source, num_workers=workers)
    source_dataloader_test = DataLoader(source_dataset_test, shuffle=True, batch_size=BATCH_SIZE_source, num_workers=workers)

    # target datasets and dataloaders
    target_dataset_train_weak = torchvision.datasets.USPS(download_path, train=True, download=True, transform=train_transform_weak)
    target_dataset_train_strong = torchvision.datasets.USPS(download_path, train=True, download=True, transform=train_transform_strong)
    target_dataset_test = torchvision.datasets.USPS(download_path, train=False, download=True, transform=test_transform)

    target_dataloader_train_weak = DataLoader(target_dataset_train_weak, shuffle=False, batch_size=BATCH_SIZE_target, num_workers=workers)
    target_dataloader_train_strong = DataLoader(target_dataset_train_strong, shuffle=False, batch_size=BATCH_SIZE_target, num_workers=workers)
    target_dataloader_test = DataLoader(target_dataset_test, shuffle=True, batch_size=BATCH_SIZE_target, num_workers=workers)

    return (source_dataloader_train_weak, source_dataloader_train_strong, source_dataloader_test), (target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test) 
