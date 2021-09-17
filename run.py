from data import get_dataloaders
from network import Encoder, Classifier
from hyperparameters import adamatch_hyperparams
from adamatch import Adamatch

# get source and target data
data = get_dataloaders("./", batch_size_source=32, workers=2)
source_dataloader_train_weak, source_dataloader_train_strong, source_dataloader_test = data[0]
target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test = data[1]

# instantiate the network
n_classes = 10
encoder = Encoder()
classifier = Classifier(n_classes=n_classes)

# instantiate AdaMatch algorithm and setup hyperparameters
adamatch = Adamatch(encoder, classifier)
hparams = adamatch_hyperparams()
epochs = 500 # my implementations uses early stopping
save_path = "./adamatch_checkpoint.pt"

# train the model
adamatch.train(source_dataloader_train_weak, source_dataloader_train_strong,
               target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test,
               epochs, hparams, save_path)

# evaluate the model
adamatch.plot_metrics()

# returns accuracy on the test set
print(f"accuracy on test set = {adamatch.evaluate(target_dataloader_test)}")

# returns a confusion matrix plot and a ROC curve plot (that also shows the AUROC)
adamatch.plot_cm_roc(target_dataloader_test)