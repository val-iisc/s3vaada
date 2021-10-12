import numpy as np
import random
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import optimizer_scheduler
import train_test

import wandb

import copy
from copy import deepcopy
from collections import OrderedDict
from sys import stderr

from torch import Tensor


class ConditionalEntropyLoss(nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0*b.mean(dim=0)


class VAT(nn.Module):
    def __init__(self, model, reduction='mean'):
        super(VAT, self).__init__()
        self.n_power = 1
        self.XI = 1e-6
        self.model = model
        self.epsilon = 5
        self.reduction = reduction

    def forward(self, X, logit, domain, lamda):
        vat_loss, r_vadv = self.virtual_adversarial_loss(
            X, logit, domain, lamda)
        return vat_loss, r_vadv

    def generate_virtual_adversarial_perturbation(self, x, logit, domain, lamda):
        d = torch.randn_like(x, device='cuda')

        for _ in range(self.n_power):
            d = self.XI * self.get_normalized_vector(d).requires_grad_()
            logit_m, _, _ = self.model(x + d, domain, lamda)
            dist = self.kl_divergence_with_logit(logit, logit_m)
            if self.reduction == 'mean':
                grad = torch.autograd.grad(dist, [d])[0]
                d = grad.detach()

        return self.epsilon * self.get_normalized_vector(d)

    def kl_divergence_with_logit(self, q_logit, p_logit):
        q = F.softmax(q_logit, dim=1)
        if self.reduction == 'mean':
            qlogq = torch.mean(
                torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
            qlogp = torch.mean(
                torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
        else:
            qlogq = torch.sum(q*F.log_softmax(q_logit, dim=1), dim=1)
            qlogp = torch.sum(q*F.log_softmax(p_logit, dim=1), dim=1)
        return qlogq - qlogp

    def get_normalized_vector(self, d):
        return F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())

    def virtual_adversarial_loss(self, x, logit, domain, lamda):
        r_vadv = self.generate_virtual_adversarial_perturbation(
            x, logit, domain, lamda)
        logit_p = logit.detach()
        logit_m, _, _ = self.model(x + r_vadv, domain, lamda)
        loss = self.kl_divergence_with_logit(logit_p, logit_m)
        return loss, r_vadv


def train(net, class_loss, domain_loss, source_dataloader,
          target_dataloader, new_data_loader, source_test_dataloader, target_test_dataloader, optimizer_, cycle, model_root, args, device):

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    fg_optimizer, fc_optimizer, dc_optimizer = optimizer_

    if new_data_loader is not None:
        new_source_dataloader = [new_data_loader, source_dataloader]
    else:
        new_source_dataloader = [source_dataloader]

    len_dataloader = min(len(source_dataloader), len(target_dataloader))
    log_frequency = int(0.5*len_dataloader)

    cent = ConditionalEntropyLoss().to(device)
    vat_loss = VAT(net).to(device)

    device = torch.device(device)

    for epoch in range(args.epochs):

        start_time = time.time()
        net.train()

        if args.log_results:
            wandb.log({'Epoch': epoch+1})

        print("Epoch :", epoch+1)

        for batch_idx, (source, target) in enumerate(zip(itertools.chain.from_iterable(new_source_dataloader), target_dataloader)):
            # Setup hyperparameters
            p = float(batch_idx + epoch * len_dataloader) / \
                (args.epochs * len_dataloader)

            lamda = 2. / (1. + np.exp(-args.gamma * p)) - 1

            # Get data input along with corresponding label
            source_input, source_label = source
            target_input, target_label = target

            if args.method == 'dann':
                fc_optimizer = optimizer_scheduler(fc_optimizer, p)
                fg_optimizer = optimizer_scheduler(fg_optimizer, p)
                dc_optimizer = optimizer_scheduler(dc_optimizer, p)

            source_input, source_label = source_input.type(torch.FloatTensor).to(
                device), source_label.type(torch.LongTensor).to(device)
            target_input, target_label = target_input.type(torch.FloatTensor).to(
                device), target_label.type(torch.LongTensor).to(device)

            fg_optimizer.zero_grad()
            fc_optimizer.zero_grad()
            dc_optimizer.zero_grad()

            domain_source_labels = torch.ones(
                source_input.shape[0], device=device, dtype=torch.long)
            domain_target_labels = torch.zeros(
                target_input.size()[0], device=device, dtype=torch.long)

            domain_target_labels_new = torch.ones(
                target_input.size()[0], dtype=torch.long, device=device)
            domain_source_labels_new = torch.zeros(
                source_input.size()[0], device=device, dtype=torch.long)

            domain_of_batch = 'source'

            if new_data_loader is not None:
                if batch_idx < len(new_data_loader):
                    domain_of_batch = 'target'
                else:
                    domain_of_batch = 'source'

            if args.method == 'vaada':
                # Method is VAADA
                with torch.cuda.amp.autocast(args.use_amp):
                    source_class_output, source_domain_output, source_features = net(
                        source_input, domain_of_batch, lamda)
                    source_class_loss = class_loss(
                        source_class_output, source_label)

                    target_class_output, target_domain_output, target_features = net(
                        target_input, 'target', lamda)
                    loss_target_cent = cent(target_class_output)

                    source_domain_loss = domain_loss(
                        source_domain_output, domain_source_labels)
                    target_domain_loss = domain_loss(
                        target_domain_output, domain_target_labels)

                    source_domain_output_d = net.domain_classifier(
                        source_features.detach())
                    target_domain_output_d = net.domain_classifier(
                        target_features.detach())

                    domain_loss_total = (domain_loss(source_domain_output_d, domain_source_labels_new) +
                                         domain_loss(target_domain_output_d, domain_target_labels_new))
                    source_loss_vat, _ = vat_loss(
                        source_input, source_class_output, domain_of_batch, lamda)
                    target_loss_vat, _ = vat_loss(
                        target_input, target_class_output, 'target', lamda)

                dc_optimizer.zero_grad()
                scaler.scale(domain_loss_total).backward()

                scaler.unscale_(dc_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    net.domain_classifier.parameters(), args.clip_value)
                scaler.step(dc_optimizer)

                total_domain_loss = 1*(source_domain_loss + target_domain_loss)

                loss = source_class_loss + 0.01*total_domain_loss + 0.01 * \
                    loss_target_cent + source_loss_vat + 0.01*target_loss_vat

                fg_optimizer.zero_grad()
                fc_optimizer.zero_grad()
                scaler.scale(loss).backward()

                scaler.unscale_(fg_optimizer)
                scaler.unscale_(fc_optimizer)

                torch.nn.utils.clip_grad_norm_(
                    net.feature_extractor.parameters(), args.clip_value)
                torch.nn.utils.clip_grad_norm_(
                    net.feature_classifier.parameters(), args.clip_value)

                scaler.step(fg_optimizer)
                scaler.step(fc_optimizer)

            else:
                #Method is DANN
                with torch.cuda.amp.autocast(args.use_amp):
                    source_class_output, source_domain_output, _ = net(
                        source_input, domain_of_batch, lamda)
                    source_class_loss = class_loss(
                        source_class_output, source_label)

                    target_features, target_domain_output, _ = net(
                        target_input, 'target', lamda)

                    source_domain_loss = domain_loss(
                        source_domain_output, domain_source_labels)
                    target_domain_loss = domain_loss(
                        target_domain_output, domain_target_labels)

                    loss = source_class_loss + source_domain_loss + target_domain_loss

                fg_optimizer.zero_grad()
                fc_optimizer.zero_grad()
                dc_optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.unscale_(fg_optimizer)
                scaler.unscale_(fc_optimizer)
                scaler.unscale_(dc_optimizer)

                torch.nn.utils.clip_grad_norm_(
                    net.feature_extractor.parameters(), args.clip_value)
                torch.nn.utils.clip_grad_norm_(
                    net.feature_classifier.parameters(), args.clip_value)
                torch.nn.utils.clip_grad_norm_(
                    net.domain_classifier.parameters(), args.clip_value)

                scaler.step(fg_optimizer)
                scaler.step(fc_optimizer)
                scaler.step(dc_optimizer)

            if args.use_amp:
                scaler.update()

            if(batch_idx % args.log_interval == 0) and args.log_results:
                wandb.log({'Source Class Loss': source_class_loss,
                           "Source Domain Loss": source_domain_loss,
                           "Target Domain Loss": target_domain_loss})

            if (batch_idx + 1) % log_frequency == 0:
                print('[{}/{} ({:.0f}%)]Source Class Loss: {:.6f}\tSource Domain Loss: {:.6f}\tTarget Domain Loss: {:.6f}'.format(
                    batch_idx *
                    args.batch_size, (len_dataloader*args.batch_size),
                    100. * batch_idx / len_dataloader,
                    source_class_loss.item(),
                    source_domain_loss.item(), target_domain_loss.item()
                ))

        avg_target_class_loss = 0
        if new_data_loader is not None:
            net.eval()
            len_new_data_loader = len(new_data_loader)
            with torch.no_grad():
                for batch_idx, target in enumerate(new_data_loader):
                    p = float(batch_idx) / (len_new_data_loader)
                    lamda = 0  # 2. / (1. + np.exp(-args.gamma * p)) - 1

                    target_input, target_label = target
                    target_input, target_label = target_input.type(torch.FloatTensor).to(
                        device), target_label.type(torch.LongTensor).to(device)
                    target_class_output, _, _ = net(
                        target_input, 'target', lamda)

                    target_class_loss = class_loss(
                        target_class_output, target_label)

                    avg_target_class_loss += target_class_loss
                print("Target Class Loss : ",
                      avg_target_class_loss.item()/len(new_data_loader))

        end_time = time.time()
        print('Time taken :', end_time-start_time)
        if (epoch+1) % 10 == 0 or args.epochs-epoch <= 50:
            cur_state = torch.get_rng_state()
            train_test.test(net, source_test_dataloader, target_test_dataloader,
                            epoch, cycle, args, device, epoch == (args.epochs-1))
            torch.set_rng_state(cur_state)
