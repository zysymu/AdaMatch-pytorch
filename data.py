from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from RandAugment import RandAugment # https://github.com/ildoonet/pytorch-randaugment
import torch

def _get_transforms(use_randaugment=False):
    """
    if `use_randugment`=True we'll make use of RandAugment some modifications to the images are necessary:
    we need to tile the images to expand them into having 3 channels and convert them into PIL images.
    """
    if use_randaugment:
        class Tile():
            def __init__(self):
                pass

            def __call__(self, tensor):
                return torch.tile(tensor, (3, 1, 1)) # copies 2d array to the other 3 channels

        train_transform_weak = transforms.Compose([transforms.ToTensor(),
                                                   Tile(),
                                                   transforms.ToPILImage(),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.Resize(32),
                                                   transforms.ToTensor()
                                                   ])

        train_transform_strong = transforms.Compose([transforms.ToTensor(),
                                                     Tile(),
                                                     transforms.ToPILImage(),
                                                     transforms.Resize(32),
                                                     RandAugment(2, 5), 
                                                     transforms.ToTensor()
                                                     ])


        test_transform = transforms.Compose([transforms.ToTensor(),
                                             Tile(),
                                             transforms.ToPILImage(),
                                             transforms.Resize(32),
                                             transforms.ToTensor()
                                             ])
                                                
    else:
        train_transform_weak = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(40),
                                                   #transforms.RandomHorizontalFlip(),
                                                   transforms.RandomCrop(32),
                                                   #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                   ])

        train_transform_strong = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize(40),
                                                     #transforms.RandomHorizontalFlip(),
                                                     #transforms.RandomVerticalFlip(),
                                                     transforms.RandomCrop(32),
                                                     transforms.RandomInvert(),
                                                     #transforms.RandomAutocontrast(),
                                                     transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
                                                     transforms.RandomErasing(),
                                                     transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.2, 1.2))
                                                     #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                     ])


        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize(32),
                                             #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])  

    return train_transform_weak, train_transform_strong, test_transform

def get_dataloaders(download_path, batch_size_source=32, workers=0, use_randaugment=False):
    train_transform_weak, train_transform_strong, test_transform = _get_transforms(use_randaugment)

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