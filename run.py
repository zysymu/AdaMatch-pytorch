import torch
from data import get_dataloaders
from network import Network
from adamatch import train, evaluate
from metrics import plot_metrics, plot_cm 

# model parameters
n_classes = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get source and target data
data = get_dataloaders("./", batch_size_source=32, workers=0)
source_dataloader_train_weak, source_dataloader_train_strong, source_dataloader_test = data[0]
target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test = data[1]

# instantiate the network
classifier = Network(pretrained=False, input_dim=1, n_classes=n_classes).to(device)

# train the network
epochs = 100
checkpoint_path = "./adamatch_checkpoint.pt"
save_path = checkpoint_path # not possible if you can't modify where `checkpoint_path` is saved

classifier, history = train(classifier,
                            source_dataloader_train_weak, source_dataloader_train_strong, target_dataloader_train_weak, target_dataloader_train_strong,
                            source_dataloader_test, target_dataloader_test,
                            epochs, device, n_classes, checkpoint_path, save_path)

# evaluate the network
print(f"\nAccuracy on source dataset: {evaluate(classifier, source_dataloader_test, device)}")
print(f"\nAccuracy on target dataset: {evaluate(classifier, target_dataloader_test, device)}")

# plot metrics
plot_metrics(history)
plot_cm(classifier, target_dataloader_test, device)