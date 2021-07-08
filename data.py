from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader

def _get_transforms():
    train_transform_weak = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(40),
                                            transforms.RandomHorizontalFlip(),
                                            #transforms.RandomVerticalFlip(),
                                            transforms.RandomCrop(32)
                                            ])

    # ideally the strong transforms would use RandAugment, but I couldn't get it working...
    train_transform_strong = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize(40),
                                                transforms.RandomHorizontalFlip(),
                                                #transforms.RandomVerticalFlip(),
                                                transforms.RandomCrop(32),
                                                transforms.RandomInvert(),
                                                transforms.RandomAutocontrast(),
                                                transforms.RandomPerspective(),
                                                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1)),
                                                transforms.RandomAffine(degrees=10, translate=(0.3, 0.3), scale=(0.7, 1.3), shear=(-2, 2, -2, 2))
                                                ])


    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(32)
                                        ])

    return train_transform_weak, train_transform_strong, test_transform

def get_dataloaders(download_path, batch_size_source=32, workers=0):
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