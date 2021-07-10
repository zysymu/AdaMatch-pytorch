import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

from adamatch import evaluate

def plot_metrics(history):
    # plot metrics for losses n stuff
    fig, axs = plt.subplots(1, 3, figsize=(18,5), dpi=200)

    epochs = len(history['epoch_loss'])

    axs[0].plot(range(1, epochs+1), history['epoch_loss'])
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Entropy loss')

    axs[1].plot(range(1, epochs+1), history['accuracy_source'])
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Accuracy on weakly augmented source')

    axs[2].plot(range(1, epochs+1), history['accuracy_target'])
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Counts')
    axs[2].set_title('Accuracy on weakly augmented target')      
        
    plt.show()

def plot_cm(classifier, dataloader, device):
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    classifier.eval()

    accuracy, labels_list, outputs_list, preds_list = evaluate(classifier, dataloader, device, return_lists_roc=True)

    # plot confusion matrix
    cm = confusion_matrix(labels_list, preds_list)

    plt.figure(figsize=(4,4), dpi=200)
    sns.heatmap(cm, cmap=cmap, fmt="")
    plt.title("Confusion matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()