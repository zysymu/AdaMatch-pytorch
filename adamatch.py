import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from sklearn.metrics import accuracy_score
import numpy as np
import os

from utils import compute_source_loss, compute_target_loss

def train(classifier, 
          source_dataloader_weak, source_dataloader_strong, target_dataloader_weak, target_dataloader_strong,
          test_source_dataloader, test_target_dataloader,
          epochs, device, n_classes, checkpoint_path, save_path):
    # configure hyperparameters (according to the paper)
    lr = 3e-3 # original: lr = 3e-2
    weight_decay = 5e-4
    tau = 0.9
    
    iters = min(len(source_dataloader_weak.dataset), len(source_dataloader_strong.dataset),
                len(target_dataloader_weak.dataset), len(target_dataloader_strong.dataset)
                )
    
    # mu related stuff
    steps_per_epoch = iters // source_dataloader_weak.batch_size
    total_steps = epochs * steps_per_epoch

    # train with checkpoints: allows training for much longer even with runtime disconnects
    if not os.path.isfile(checkpoint_path): # if checkpoint doesn't exist
        optimizer = optim.Adam(list(classifier.parameters()), lr=lr, weight_decay=weight_decay)
        #scheduler = CosineAnnealingWarmRestarts(optimizer, total_steps, eta_min=lr*0.25)
        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

        start_epoch = 0
        current_step = 0

        history = {'epoch_loss': [], 'accuracy_source': [], 'accuracy_target': []}

    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        classifier.load_state_dict(checkpoint['classifier_weights'])

        optimizer = optim.Adam(list(classifier.parameters()), lr=lr, weight_decay=weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_weights'])

        #scheduler = CosineAnnealingWarmRestarts(optimizer, total_steps, eta_min=lr*0.25)
        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
        scheduler.load_state_dict(checkpoint['scheduler_weights'])

        start_epoch = checkpoint['epoch'] + 1
        current_step = checkpoint['step']
            
        history = checkpoint['history']    

    # training loop
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0

        # set network to training mode
        classifier.train()

        dataset = zip(source_dataloader_weak, source_dataloader_strong,
                      target_dataloader_weak, target_dataloader_strong
                      )

        # this is where the unsupervised learning comes in, as such, we're not interested in labels
        for (data_source_weak, labels_source), (data_source_strong, _), (data_target_weak, _), (data_target_strong, _) in dataset:
            data_source_weak = data_source_weak.to(device)
            labels_source = labels_source.to(device)

            data_source_strong = data_source_strong.to(device)
            data_target_weak = data_target_weak.to(device)
            data_target_strong = data_target_strong.to(device)

            # concatenate data (in case of low GPU power this could be done after classifying with the model)
            data_combined = torch.cat([data_source_weak, data_source_strong, data_target_weak, data_target_strong], 0)
            source_combined = torch.cat([data_source_weak, data_source_strong], 0)

            # get source data limit (useful for slicing later)
            source_total = source_combined.size(0)

            # zero gradients
            optimizer.zero_grad()

            # forward pass: calls the model once for both source and target and once for source only
            logits_combined = classifier(data_combined)
            logits_source_p = logits_combined[:source_total]

            classifier.eval() # doesn't update certain layers
            logits_source_pp = classifier(source_combined)
            classifier.train()

            # perform random logit interpolation
            lambd = torch.rand(source_total, n_classes).to(device)
            final_logits_source = (lambd * logits_source_p) + ((1-lambd) * logits_source_pp)

            # distribution allignment
            ## softmax for logits of weakly augmented source images
            logits_source_weak = final_logits_source[:data_source_weak.size(0)]
            pseudolabels_source = F.softmax(logits_source_weak, 0)

            ## softmax for logits of weakly augmented target images
            logits_target = logits_combined[source_total:]
            logits_target_weak = logits_target[:data_target_weak.size(0)]
            pseudolabels_target = F.softmax(logits_target_weak, 0)

            ## allign target label distribtion to source label distribution
            expectation_ratio = torch.mean(pseudolabels_source) / torch.mean(pseudolabels_target)
            final_pseudolabels = F.normalize((pseudolabels_target*expectation_ratio), p=2, dim=1) # L2 normalization

            # perform relative confidence thresholding
            row_wise_max, _ = torch.max(pseudolabels_source, dim=-1)
            final_sum = torch.mean(row_wise_max, 0)
            
            ## define relative confidence threshold
            c_tau = tau * final_sum

            max_values, _ = torch.max(final_pseudolabels, dim=-1)
            mask = max_values >= c_tau

            # compute loss
            source_loss = compute_source_loss(logits_source_weak, final_logits_source[data_source_weak.size(0):], labels_source)
            
            final_pseudolabels = torch.max(final_pseudolabels,1)[1]
            target_loss = compute_target_loss(final_pseudolabels.long(), logits_target[data_target_weak.size(0):], mask)

            ## compute target loss weight (mu)
            pi = torch.tensor(np.pi, dtype=torch.float).to(device)
            step = torch.tensor(current_step, dtype=torch.float).to(device)
            mu = 0.5 - torch.cos(torch.minimum(pi, (2*pi*step) / total_steps)) / 2

            ## get total loss
            loss = source_loss + (mu * target_loss)
            current_step += 1

            # backpropagate and update weights
            loss.backward()
            optimizer.step()

            # metrics
            running_loss += loss.item()

            # cosine scheduler step
            #scheduler.step()

        # get losses
        # we use np.min because zip only goes up to the smallest list length
        epoch_loss = running_loss / iters

        history['epoch_loss'].append(epoch_loss)

        # evaluate on test data
        epoch_accuracy_source = evaluate(classifier, test_source_dataloader, device)
        epoch_accuracy_target = evaluate(classifier, test_target_dataloader, device)
        history['accuracy_source'].append(epoch_accuracy_source)
        history['accuracy_target'].append(epoch_accuracy_target)

        # steplr scheduler step
        scheduler.step()

        # save checkpoint
        torch.save({'classifier_weights': classifier.state_dict(),
                    'optimizer_weights': optimizer.state_dict(),
                    'scheduler_weights': scheduler.state_dict(),
                    'epoch': epoch,
                    'step': current_step,
                    'history': history}, save_path)

        print('[Epoch {}/{}] loss: {:.6f}; accuracy source: {:.6f}; accuracy target: {:.6f}'.format(epoch+1, epochs, epoch_loss, epoch_accuracy_source, epoch_accuracy_target))

    return classifier, history


def evaluate(classifier, dataloader, device, return_lists_roc=False):
    # set network to evaluation mode
    classifier.eval()

    labels_list = []
    outputs_list = []
    preds_list = []

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)

            # predict
            outputs = F.softmax(classifier(data), dim=1)

            # numpify
            labels_numpy = labels.detach().cpu().numpy()
            outputs_numpy = outputs.detach().cpu().numpy() # probs (AUROC)
            
            preds = np.argmax(outputs_numpy, axis=1) # accuracy

            # append
            labels_list.append(labels_numpy)
            outputs_list.append(outputs_numpy)
            preds_list.append(preds)

        labels_list = np.concatenate(labels_list)
        outputs_list = np.concatenate(outputs_list)
        preds_list = np.concatenate(preds_list)

    # metrics
    accuracy = accuracy_score(labels_list, preds_list)

    if return_lists_roc:
        return accuracy, labels_list, outputs_list, preds_list
        
    return accuracy