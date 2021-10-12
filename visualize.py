import train_test
import wandb
import torch.nn as nn
import torch
import seaborn as sns
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import os
from sklearn import manifold
import matplotlib
matplotlib.use('Agg')


def new_TSNE(net, source_dataloader, target_dataloader, new_data_loader, temp_dataloader, cycle, device, args):

    net.eval()
    source_embedding = torch.tensor([]).to(device)
    source_labels = torch.tensor([]).type(torch.LongTensor)  # .to(device)

    f = plt.figure(figsize=(16, 16))
    lamda = 0
    with torch.no_grad():
        for batch_idx, source_data in enumerate(source_dataloader):
            source_input, source_label = source_data
            source_input = source_input.to(device)
            source_feature = net.feature_extractor(
                source_input, 'source', lamda)

            source_embedding = torch.cat((source_embedding, source_feature), 0)
            source_labels = torch.cat((source_labels, source_label), 0)

    target_embedding = torch.tensor([]).to(device)
    target_labels = torch.tensor([]).type(torch.LongTensor)  # .to(device)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(target_dataloader):
            p = float(batch_idx) / len(target_dataloader)
            inputs = inputs.to(device)

            target_feature = net.feature_extractor(inputs, 'target', lamda)
            target_embedding = torch.cat((target_embedding, target_feature), 0)
            target_labels = torch.cat((target_labels, labels), 0)

    newly_labeled_embedding = torch.tensor([]).to(device)
    new_labels = torch.tensor([]).type(torch.LongTensor)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(temp_dataloader):
            inputs = inputs.to(device)
            target_feature = net.feature_extractor(inputs, 'target', lamda)
            newly_labeled_embedding = torch.cat(
                (newly_labeled_embedding, target_feature), 0)
            new_labels = torch.cat((new_labels, labels), 0)

    labeled_embedding = torch.tensor([]).to(device)
    old_labels = torch.tensor([]).type(torch.LongTensor)
    with torch.no_grad():
        if new_data_loader is not None:
            for batch_idx, (inputs, labels) in enumerate(new_data_loader):
                inputs = inputs.to(device)
                target_feature = net.feature_extractor(inputs, 'target', lamda)
                labeled_embedding = torch.cat(
                    (labeled_embedding, target_feature), 0)
                old_labels = torch.cat((old_labels, labels), 0)

    source_embedding = source_embedding.cpu().numpy()
    labeled_embedding = labeled_embedding.cpu().numpy()
    target_embedding = target_embedding.cpu().numpy()
    newly_labeled_embedding = newly_labeled_embedding.cpu().numpy()

    source_labels = source_labels.cpu().numpy()
    target_labels = target_labels.cpu().numpy()
    new_labels = new_labels.cpu().numpy()
    old_labels = old_labels.cpu().numpy()

    if new_data_loader is None:
        X = np.concatenate(
            (source_embedding, target_embedding, newly_labeled_embedding), axis=0)
    else:
        X = np.concatenate((source_embedding, target_embedding,
                           newly_labeled_embedding, labeled_embedding), axis=0)
    tsne = TSNE(n_jobs=8)
    X_tsne = tsne.fit_transform(X)

    source_embedding = X_tsne[:len(source_embedding)]
    target_embedding = X_tsne[len(source_embedding):len(
        source_embedding)+len(target_embedding)]

    newly_labeled_embedding = X_tsne[len(source_embedding)+len(target_embedding):len(
        source_embedding)+len(target_embedding)+len(newly_labeled_embedding)]
    if new_data_loader is not None:
        labeled_embedding = X_tsne[len(
            source_embedding)+len(target_embedding)+len(newly_labeled_embedding):]

    n_class = args.num_classes
    palette = np.array(sns.color_palette('hls', n_class))

    plt.scatter(source_embedding[:, 0], source_embedding[:, 1], lw=0, s=20,
                c=palette[source_labels.astype(np.int)], marker='o')  # , alpha=0.3)
    plt.scatter(target_embedding[:, 0], target_embedding[:, 1], lw=0, s=20,
                c=palette[target_labels.astype(np.int)], marker='*')  # , alpha=0.7)
    #plt.plot(newly_labeled_embedding[:, 0], newly_labeled_embedding[:, 1],linestyle='none',markersize=100, markeredgecolor="orange", markeredgewidth=10)
    plt.scatter(newly_labeled_embedding[:, 0], newly_labeled_embedding[:, 1], s=80, c=palette[new_labels.astype(
        np.int)], marker='s', edgecolor='red', linewidths=3)  # , alpha=0.5)
    if new_data_loader is not None:
        plt.scatter(labeled_embedding[:, 0], labeled_embedding[:, 1], lw=0, s=90,
                    c=palette[old_labels.astype(np.int)], marker='>')  # , alpha=0.5)

    if args.log_results:
        wandb.log({f"Cycle{cycle+1}": plt})

    plt.close()


def analyze(idx, target_dataset, net, args, device):

    print("The queried samples belong to the following classes: ")
    for index in idx:
        print(target_dataset[index][1], end=' ')
        # classes_of_new_samples = torch.cat((classes_of_new_samples,torch.from_numpy(target_dataset[index][1])),0)
    print()


if __name__ == '__main__':

    X = np.random.randn(1000, 50)
    tsne = TSNE(n_jobs=4)
    Y = tsne.fit_transform(X)
    print()
