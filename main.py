import argparse
import datasets
import errno
import models
import torch

import train_test
import active_learning as al
from visualize import new_TSNE, analyze

import os
from datetime import datetime
import numpy as np
import torch.optim as optim

import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_CHECKPOINTS = CURRENT_DIR_PATH + '/models/models_checkpoints/'

# dd/mm/YY H:M:S
time_stamp = datetime.now().strftime("%d%m%Y_%H%M%S")


def make_args_parser():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description='Active Domain Adaptation via S3VAADA')
    # fill parser with information about program arguments
    parser.add_argument('-s', '--source', default='webcam', type=str,
                        help='Define the source domain')
    parser.add_argument('-t', '--target', default='amazon', type=str,
                        help='Define the target domain')
    parser.add_argument('-m', '--model', default='ResNet', type=str,
                        help='Define the architecture')
    parser.add_argument('-bs', '--batch_size', default=36, type=int,
                        help='Batch Size')
    parser.add_argument('-c', '--cycles', default=6, type=int,
                        help='Number of Cycles')
    parser.add_argument('-e', '--epochs', default=100, type=int,
                        help='Number of Epochs')
    parser.add_argument('-k', '--learning_rate', default=1e-2, type=float,
                        help='Learning rate')
    parser.add_argument('-w', '--workers', default=4, type=int,
                        help='Number of workers')
    parser.add_argument('-al', '--sampling', default='s3vaada', type=str,
                        help='Sampling Strategy for active learning')
    parser.add_argument('-im', '--image_size', default=224, type=int,
                        help='Image Size')
    parser.add_argument('-mo', '--momentum', default=0.9, type=float,
                        help='Momentum')
    parser.add_argument('-wd', '--weight_decay', default=0.0005, type=float,
                        help='weight decay for SGD')
    parser.add_argument('-se', '--seed', default=123, type=int,
                        help='Seed for the run')
    parser.add_argument('-met', '--method', default="vaada", type=str,
                        help='Method : dann or vaada')
    parser.add_argument('-clip', '--clip_value', default=1, type=float,
                        help='Clip value for max norm')
    parser.add_argument('-g', '--gamma', default=10, type=float,
                        help='Gamma value in the schedule (as defined in DANN)')
    parser.add_argument('-log', '--log_interval', default=50, type=int,
                        help='Log interval for wandb')
    parser.add_argument('-na', '--name', default="test", type=str,
                        help='Wandb name run')
    parser.add_argument('-amp', '--use_amp', default=True, type=bool,
                        help='Mixed Precision Training')
    parser.add_argument('-logr', '--log_results', default=True, type=bool,
                        help='To log results or not')
    parser.add_argument('-gid', '--gpu', default=1, type=int,
                        help='GPU to use')
    parser.add_argument('-a', '--alpha', default=0.5, type=float,
                        help="alpha value for submodular function")
    parser.add_argument('-b', '--beta', default=0.3, type=float,
                        help="beta value for submodular function")
    parser.add_argument('-r', '--resume', default="", type=str,
                        help="Resume from checkpoint")
    parser.add_argument('-bud', '--budget', default=None, type=int,
                        help='Budget to use')
    return parser.parse_args()


def print_args(args):
    print("Running with the following configuration")

    args_map = vars(args)
    for key in args_map:
        print('\t', key, '-->', args_map[key])
    print()


def main():
    # parse and print arguments
    args = make_args_parser()
    print_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Check device available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on: {}".format(device))

    # Seed Everything
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # Timestamp
    args.time_stamp = time_stamp

    # Load both source and target domain datasets
    source_dataloader, source_dataset = datasets.get_source_domain(
        args.source, args)
    source_test_dataloader, _ = datasets.get_source_domain(
        args.source, args, train=False)

    (target_dataset, target_dataloader), (test_dataset,
                                          target_test_dataloader) = datasets.get_target_domain(args.target, args)

    # Set Budget as 2% of the number of samples in the target dataset
    if args.budget is None:
        args.budget = int(len(target_dataset)*0.02)

    print("Budget for every cycle : ", args.budget)
    # Create directory to save model's checkpoints
    try:
        model_root = MODEL_CHECKPOINTS + args.source + '-' + \
            args.target + "/" + args.sampling + "/" + args.time_stamp + "/"
        print("Model saved at = ", model_root)
        os.makedirs(model_root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    # Intialize Wandb
    if args.log_results:
        wandb.init(project="active-learning",
                   entity="active-learning", name=args.name)
        wandb.config.update(args)
        wandb.config.update({"Optimizer": "SGD"})

    # Initialize model

    net = models.ResNet(args.num_classes, device, args)
    param_dict = torch.load('models/resnet50.pth')
    models.load_single_state_dict(net, param_dict)

    net = net.to(device)

    domain_loss = torch.nn.CrossEntropyLoss()
    class_loss = torch.nn.CrossEntropyLoss()

    if args.log_results:
        wandb.watch(net)
    torch.save(net.state_dict(), model_root + "/" + args.name + ".pth")
    cycle_no = 0

    if args.resume:
        last_cycle_weight = sorted([x for x in os.listdir(args.resume) if x.endswith(
            ".pth") and len(x.strip(".pth")) < 3], key=lambda x: (len(x), x))[-2]
        print("Resuming from checkpoint:", last_cycle_weight)
        net.load_state_dict(torch.load(
            os.path.join(args.resume, last_cycle_weight)))
        cycle_no = last_cycle_weight.strip(".pth")  # [0]
        all_idx = np.array([])
        for i in range(int(cycle_no)+1):
            idx = np.load(os.path.join(args.resume, str(i)+".npy"))
            all_idx = np.concatenate((all_idx, idx))
        all_idx = all_idx.astype(int)
        all_idx = torch.from_numpy(all_idx)
        all_indices = torch.arange(0, len(target_dataset))
        new_data_set = torch.utils.data.Subset(target_dataset, all_idx)
        target_dataset = torch.utils.data.Subset(target_dataset, torch.from_numpy(
            np.setdiff1d(all_indices.numpy(), all_idx.numpy())))
        target_dataloader = DataLoader(
            dataset=target_dataset,
            batch_size=args.batch_size, num_workers=args.workers,
            shuffle=True
        )
        new_data_loader = DataLoader(
            dataset=new_data_set,
            batch_size=args.batch_size, num_workers=args.workers,
            shuffle=True
        )
        print("Number of labeled target samples:", len(all_idx))
        cycle_no = int(cycle_no)+1

    print("Number of classes: ", args.num_classes)
    print("Number of images in the target dataset : ", len(target_dataset))
    print("Number of images in the source dataset : ", len(source_dataset))

    new_data_loader = None

    for cycle in range(cycle_no, args.cycles):

        print('Cycle: ', cycle+1)
        if args.log_results:
            wandb.log({"Cycle": cycle+1})

        # Load the original ResNet-50 weights
        net.load_state_dict(torch.load(model_root + "/" + args.name + ".pth"))

        dc_optimizer = optim.SGD(net.domain_classifier.parameters(
        ), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

        if args.method == "dann":
            fg_optimizer = optim.SGD(net.feature_extractor.parameters(
            ), lr=args.learning_rate/10, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            fg_optimizer = optim.SGD(net.feature_extractor.parameters(
            ), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

        fc_optimizer = optim.SGD(net.feature_classifier.parameters(
        ), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

        train_test.train(net, class_loss, domain_loss, source_dataloader,
                         target_dataloader, new_data_loader, source_test_dataloader, target_test_dataloader,
                         (fg_optimizer, fc_optimizer, dc_optimizer),
                         cycle, model_root, args, device)

        # To sample the images from the unlabeled target dataset
        unshuffled_dataloader = DataLoader(
            dataset=target_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False
        )

        len_data_loader = len(unshuffled_dataloader.dataset)
        all_indices = torch.arange(0, len_data_loader)

        idx = al.get_active_learning_method(
            net, unshuffled_dataloader, device, args, source_dataloader, cycle, new_data_loader)
        # Displays which classes the selected samples belong to
        analyze(idx, target_dataset, net, args, device)

        temp_dataset = torch.utils.data.Subset(target_dataset, idx)
        temp_dataloader = DataLoader(
            dataset=temp_dataset,
            batch_size=args.batch_size, num_workers=args.workers,
            shuffle=False
        )
        # Visualize
        new_TSNE(net, source_dataloader, target_dataloader,
                 new_data_loader, temp_dataloader, cycle, device, args)

        if new_data_loader is None:
            new_data_set = torch.utils.data.Subset(target_dataset, idx)
        else:
            new_data_set = torch.utils.data.ConcatDataset(
                [new_data_set, torch.utils.data.Subset(target_dataset, idx)])

        # Remove the labeled images from the target dataset
        target_dataset = torch.utils.data.Subset(target_dataset, torch.from_numpy(
            np.setdiff1d(all_indices.numpy(), idx.numpy())))
        target_dataloader = DataLoader(
            dataset=target_dataset,
            batch_size=args.batch_size, num_workers=args.workers,
            shuffle=True
        )
        # new_data_loader contains the labeled target images
        new_data_loader = DataLoader(
            dataset=new_data_set,
            batch_size=args.batch_size, num_workers=args.workers,
            shuffle=True
        )
        np.save(model_root+str(cycle)+".npy", idx)


if __name__ == '__main__':
    main()
