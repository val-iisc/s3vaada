import numpy as np
import torch
import wandb
from torch.autograd import Variable
import os
import random
import copy

best_source_acc = 0
best_target_acc = 0
best_domain_acc = 0


def test(net, source_dataloader, target_dataloader, epoch, cycle, args, device, final_epoch):
    # Setup model
    net.eval()

    source_label_correct = 0.0
    target_label_correct = 0.0

    source_domain_correct = 0.0
    target_domain_correct = 0.0
    domain_correct = 0.0

    lamda = 0
    len_src_dataset = len(source_dataloader.dataset)

    # Testing on small subset of Source dataset
    small_src_dataset = torch.utils.data.Subset(source_dataloader.dataset, random.sample(
        range(0, len_src_dataset), len_src_dataset//10))
    source_dataloader = torch.utils.data.DataLoader(
        small_src_dataset, batch_size=args.batch_size,  shuffle=False, num_workers=args.workers)
    # Test source data
    with torch.no_grad():
        for batch_idx, source_data in enumerate(source_dataloader):

            source_input, source_label = source_data

            source_input, source_label = Variable(
                source_input.to(device)), Variable(source_label.to(device))
            source_labels = Variable(torch.zeros(
                (source_input.size()[0])).type(torch.LongTensor).to(device))

            source_label_pred, source_domain_pred, _ = net(
                source_input, 'source', lamda)

            source_label_pred = source_label_pred.data.max(1, keepdim=True)[1]
            source_label_correct += source_label_pred.eq(
                source_label.data.view_as(source_label_pred)).cpu().sum()

            source_domain_pred = source_domain_pred.data.max(1, keepdim=True)[
                1]
            source_domain_correct += source_domain_pred.eq(
                source_labels.data.view_as(source_domain_pred)).cpu().sum()

    # Test target data
    with torch.no_grad():
        for batch_idx, target_data in enumerate(target_dataloader):

            target_input, target_label = target_data

            target_input, target_label = target_input.type(torch.FloatTensor).to(
                device), target_label.type(torch.LongTensor).to(device)
            target_labels = Variable(torch.ones(
                (target_input.size()[0])).type(torch.LongTensor).to(device))
            # Compute target accuracy both for label and domain predictions
            target_label_pred_, target_domain_pred, _ = net(
                target_input, 'target', lamda)

            target_label_pred = target_label_pred_.data.max(1, keepdim=True)[1]
            target_label_correct += target_label_pred.eq(
                target_label.data.view_as(target_label_pred)).cpu().sum()

            target_domain_pred = target_domain_pred.data.max(1, keepdim=True)[
                1]
            target_domain_correct += target_domain_pred.eq(
                target_labels.data.view_as(target_domain_pred)).cpu().sum()

    # Compute domain correctness
    domain_correct = source_domain_correct + target_domain_correct

    global best_source_acc, best_target_acc, best_domain_acc

    target_acc = float(target_label_correct) / len(target_dataloader.dataset)
    source_acc = float(source_label_correct) / len(source_dataloader.dataset)
    domain_acc = float(domain_correct) / \
        (len(source_dataloader.dataset) + len(target_dataloader.dataset))

    if target_acc > best_target_acc:
        best_target_acc = target_acc
        best_source_acc = source_acc
        best_domain_acc = domain_acc

        CURRENT_DIR_PATH = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir))

        MODEL_CHECKPOINTS = CURRENT_DIR_PATH + '/models/models_checkpoints/'
        model_root = MODEL_CHECKPOINTS + args.source + '-' + \
            args.target + "/" + args.sampling + "/" + args.time_stamp

        PATH = model_root + "/" + str(cycle) + ".pth"
        torch.save(net.state_dict(), PATH)

    # Print results
    if args.log_results and final_epoch:
        wandb.log({"Source Accuracy": 100. * best_source_acc,
                   "Domain Accuracy": 100. * best_domain_acc,
                   "Number of labeled images": cycle*args.budget, "Target Accuracy": 100. * best_target_acc})
        # After every cycle: reset
        best_target_acc = 0
        best_source_acc = 0
        best_domain_acc = 0

    print('\nSource Accuracy: {}/{} ({:.4f}%)\nTarget Accuracy: {}/{} ({:.4f}%)\n'
          'Domain Accuracy: {}/{} ({:.4f}%)\n'.
          format(
              source_label_correct, len(source_dataloader.dataset),
              100. * source_acc,
              target_label_correct, len(target_dataloader.dataset),
              100. * target_acc,
              domain_correct, len(source_dataloader.dataset) +
              len(target_dataloader.dataset),
              100. * float(domain_correct) / (
                  len(source_dataloader.dataset) + len(
                      target_dataloader.dataset))
          ))
