import numpy as np
import torch
import train_test

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from copy import deepcopy

from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from scipy import stats


def get_active_learning_method(net, unlabeled_loader, device, args, source_dataloader, cycle, new_dataloader):

    if args.sampling == 'random':
        idx = random_sampling(net, unlabeled_loader, device, args)
    elif args.sampling == 'aada':
        idx = aada(net, unlabeled_loader, device, args)
    elif args.sampling == 'bvsb':
        idx = BvSB(net, unlabeled_loader, device, args)
    elif args.sampling == 'coreset':
        idx = coreset_sampling(net, unlabeled_loader, device, args)
    elif args.sampling == 'badge':
        idx = badge_sampling(net, unlabeled_loader, device, args)
    elif args.sampling == 's3vaada':
        idx = s3vaada(net, unlabeled_loader, device, args,
                      cycle, source_dataloader, new_dataloader)
    else:
        raise NotImplementedError()
    return idx

def H(x):
    return -1*torch.sum(torch.exp(x) * x, dim=1)


def aada(models, unlabeled_loader, device, args):

    models.eval()
    uncertainty = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(unlabeled_loader):
            p = float(batch_idx) / len(unlabeled_loader)
            lamda = 0
            inputs = inputs.to(device)
            scores, domain_scores, _ = models(inputs, 'target', lamda)

            pt = torch.exp(domain_scores[:, 1]) / \
                torch.exp(domain_scores).sum(dim=1)
            ps = torch.exp(domain_scores[:, 0]) / \
                torch.exp(domain_scores).sum(dim=1)
            weight = (1 - pt) / pt  # Importance weight
            weight = pt / ps
            # Add the entropy term to the weight
            weight = weight * H(scores)
            weight = weight.view(weight.size(0))
            uncertainty = torch.cat((uncertainty, weight), 0)

    uncertainty = torch.argsort(uncertainty, descending=True)
    idx = uncertainty.narrow(0, 0, args.budget)
    idx = idx.cpu()
    return idx


def random_sampling(net, unlabeled_loader, device, args):
    number_of_unlabeled_samples = len(unlabeled_loader.dataset)
    idx = torch.from_numpy(np.random.choice(
        number_of_unlabeled_samples, args.budget, replace=False))
    return idx


def BvSB(net, unlabeled_loader, device, args):
    net.eval()
    diff_top2 = torch.tensor([])
    lamda = 0
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(unlabeled_loader):
            p = float(batch_idx) / len(unlabeled_loader)
            inputs = inputs.to(device)
            target_class_pred, _, _ = net(inputs, 'target', lamda)
            class_prob = F.softmax(target_class_pred, dim=1)
            top2_prob, top2_pred = torch.topk(class_prob, 2)
            for i in top2_prob:
                diff_top2 = torch.cat(
                    (diff_top2, torch.tensor([(i[0]-i[1])])), 0)
    ranked = torch.argsort(diff_top2, descending=False)
    print(ranked)
    idx = ranked.narrow(0, 0, args.budget)
    return idx


class VAT(nn.Module):
    def __init__(self, model, reduction='mean'):
        super(VAT, self).__init__()
        self.n_power = 1
        self.XI = 1e-6
        self.model = model
        self.epsilon = 5.0

    def forward(self, X, logit, domain, lamda):
        vat_loss, r_vadv = self.virtual_adversarial_loss(
            X, logit, domain, lamda)
        return vat_loss, r_vadv

    def generate_virtual_adversarial_perturbation(self, x, logit, domain, lamda, random=None):
        if random is None:
            d = torch.randn_like(x, device='cuda')
        else:
            d = random
        lamda = 0
        for _ in range(self.n_power):
            d = self.XI * self.get_normalized_vector(d).requires_grad_()
            logit_m, _, _ = self.model(x + d, domain, lamda)
            dist = self.kl_divergence_with_logit(logit, logit_m)
            grad = torch.autograd.grad(dist, [d])[0]
            d = grad.detach()

        return self.epsilon * self.get_normalized_vector(d)

    def kl_divergence_with_logit(self, q_logit, p_logit, reduction="mean"):
        q = F.softmax(q_logit, dim=1)
        if reduction == 'mean':
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
        with torch.no_grad():
            logit_m, _, _ = self.model(x + r_vadv, domain, lamda)
            loss = self.kl_divergence_with_logit(
                logit_p, logit_m, reduction="none")
        return loss, (r_vadv, logit_m)


# K-Means++ utility function
def init_centers(X, K):
    # take the maximum norm one vector as c0
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    # centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        D2 = D2.ravel().astype(float)
        # SAMPLING WITH PMF = D2/sum(D2)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(
            name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
    return indsAll


def get_grad_embedding(model, unlabeled_loader, args):
    embDim = 256
    model.eval()
    nLab = args.num_classes
    embedding = np.zeros([len(unlabeled_loader.dataset), embDim * nLab])
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(unlabeled_loader):
            x, y = Variable(x.cuda()), Variable(y.cuda())
            idxs = np.arange(len(x)) + args.batch_size * batch_idx
            lamda = 0
            cout, _, out = model(x, 'target', lamda)
            out = out.data.cpu().numpy()
            batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
            maxInds = np.argmax(batchProbs, 1)
            for j in range(len(y)):
                for c in range(nLab):
                    if c == maxInds[j]:
                        embedding[idxs[j]][embDim * c: embDim *
                                           (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                    else:
                        embedding[idxs[j]][embDim * c: embDim *
                                           (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
        return torch.Tensor(embedding)


def badge_sampling(net, unlabeled_loader, device, args):
    net.eval()
    idxs_unlabeled = np.arange(len(unlabeled_loader.dataset))
    gradEmbedding = get_grad_embedding(
        net, unlabeled_loader, args).cpu().numpy()
    print("Grad embedding shape = ", gradEmbedding.shape)
    chosen = init_centers(gradEmbedding, args.budget)
    print("chosen = ", chosen)
    idxs = idxs_unlabeled[chosen]
    print("idxs = ", idxs)
    return torch.from_numpy(idxs)


def coreset_sampling(net, unlabeled_loader, device, args):
    print('Core-Set Sampling')
    net.eval()
    lamda = 0
    embedding = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(unlabeled_loader):
            p = float(batch_idx) / len(unlabeled_loader)
            inputs = inputs.to(device)
            feature = net.feature_extractor(inputs, 'target', lamda)
            embedding = torch.cat((embedding, feature), 0)

    embedding = embedding.cpu().numpy()
    number_of_unlabeled_samples = len(unlabeled_loader.dataset)

    dist_mat = np.matmul(embedding, embedding.transpose())

    sq = np.array(dist_mat.diagonal()).reshape(number_of_unlabeled_samples, 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)

    NUM_INIT_LB = 10
    idxs_lb = np.zeros(number_of_unlabeled_samples, dtype=bool)
    idxs_tmp = np.arange(number_of_unlabeled_samples)
    np.random.shuffle(idxs_tmp)
    idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

    lb_flag = idxs_lb.copy()
    mat = dist_mat[~lb_flag, :][:, lb_flag]

    for i in range(args.budget):
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(number_of_unlabeled_samples)[~lb_flag][q_idx_]
        lb_flag[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

    opt = mat.min(axis=1).max()

    bound_u = opt
    bound_l = opt/2.0
    delta = opt

    xx, yy = np.where(dist_mat <= opt)
    dd = dist_mat[xx, yy]

    lb_flag_ = idxs_lb.copy()
    subset = np.where(lb_flag_ == True)[0].tolist()

    SEED = 5

    import pickle
    #pickle.dump((xx.tolist(), yy.tolist(), dd.tolist(), subset, float(opt), args.budget, number_of_unlabeled_samples), open('mip{}.pkl'.format(SEED), 'wb'), 2)

    # import ipdb
    # ipdb.set_trace()
    # solving MIP
    # download Gurobi software from http://www.gurobi.com/
    # sh {GUROBI_HOME}/linux64/bin/gurobi.sh < core_set_sovle_solve.py

    #import os
    #os.system('sh ./gurobi902/linux64/bin/gurobi.sh < core_set_sovle_solve.py')

    #sols = pickle.load(open('sols{}.pkl'.format(SEED), 'rb'))
    sols = None

    if sols is None:
        q_idxs = lb_flag
    else:
        lb_flag_[sols] = True
        q_idxs = lb_flag_
    print('sum q_idxs = {}'.format(q_idxs.sum()))

    return torch.from_numpy(np.arange(number_of_unlabeled_samples)[(idxs_lb ^ q_idxs)])


def get_vat(net, unlabeled_loader, device, args):
    vat_loss = VAT(net, reduction='mean').to(device)
    net.eval()
    vat_loss_all = torch.tensor([]).to(device)
    restarts = 5
    lamda = 0

    for batch_idx, target in enumerate(unlabeled_loader):
        target_input, target_label = target
        target_input, target_label = target_input.type(torch.FloatTensor).to(
            device), target_label.type(torch.LongTensor).to(device)
        target_class_output, _, _ = net(target_input, 'target', lamda)
        logits = target_class_output

        vat_loss_restarts = None
        logit_batch = None
        for i in range(restarts):
            target_loss_vat, (r_vadv, logit) = vat_loss(
                target_input, logits, 'target', lamda)
            if vat_loss_restarts is None:
                vat_loss_restarts = target_loss_vat.unsqueeze(0)
                logit_batch = logit.unsqueeze(0)
            else:
                vat_loss_restarts = torch.cat(
                    (vat_loss_restarts, target_loss_vat.unsqueeze(0)), 0)
                logit_batch = torch.cat((logit_batch, logit.unsqueeze(0)), 0)
        kl_avg = vat_loss_restarts
        for i in range(restarts):
            for j in range(restarts):
                if i != j:
                    x = vat_loss.kl_divergence_with_logit(
                        logit_batch[i], logit_batch[j], reduction="none")
                    if kl_avg is None:
                        kl_avg = x.unsqueeze(0)
                    else:
                        kl_avg = torch.cat((kl_avg, x.unsqueeze(0)), 0)
        output = torch.sum(kl_avg, dim=0)
        vat_loss_all = torch.cat((vat_loss_all, output), 0)
    return vat_loss_all.cpu().numpy()/(restarts**2)


def get_embedding(net, unlabeled_loader, device, args):
    net.eval()
    lamda = 0
    embedding = torch.tensor([]).to(device)
    with torch.no_grad():
        for _, (inputs, _) in enumerate(unlabeled_loader):
            inputs = inputs.to(device)
            feature = net.feature_extractor(inputs, 'target', lamda)
            embedding = torch.cat((embedding, feature), 0)
    return embedding.cpu().numpy()


def get_softmax_output(net, unlabeled_loader, device, args):
    net.eval()
    softmax_output = torch.tensor([]).to(device)
    lamda = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(unlabeled_loader):
            p = float(batch_idx) / len(unlabeled_loader)
            inputs = inputs.to(device)
            target_class_pred, _, feat = net(inputs, 'target', lamda)
            target_class_pred = F.softmax(target_class_pred, dim=1)
            softmax_output = torch.cat((softmax_output, target_class_pred), 0)
    print('softmax_output shape = ', softmax_output.shape)
    return softmax_output.cpu().numpy()


def Gain(vat, kl, similarity, A, S, alpha=1., beta=1.):
    vat = vat[A]

    kl_score = kl[S, :][:, A].T  # + kl[A,:][:,S]
    if kl_score.shape[1] == 0:
        kl_score = np.zeros(kl_score.shape[0])
    else:
        kl_score = kl_score.min(axis=1)

    sim_score_of_all_with_selected = similarity[:, S]
    if sim_score_of_all_with_selected.shape[1] == 0:
        sim_score = np.zeros(sim_score_of_all_with_selected.shape[0])
    else:
        sim_score_of_all_with_selected = similarity[:, S].max(
            axis=1).reshape(-1, 1)
        sim_score_of_all_with_not_selected = similarity[:, A]
        sim_score_of_all = sim_score_of_all_with_not_selected - \
            sim_score_of_all_with_selected
        sim_score_of_all[sim_score_of_all < 0] = 0
        sim_score = sim_score_of_all.mean(axis=0)

    # Combining the three scores
    score = alpha*vat + beta*kl_score + (1-alpha-beta)*sim_score

    selected = score.argmax()
    print("Convex comb: VAP, KL = ",
          vat[selected], kl_score[selected], sim_score[selected])

    return score


def pairwise_kl_gpu(A, B):
    A1 = A[:, None, :]
    A2 = B[None, :, :]
    div = A1/A2
    log = torch.log(div)
    log = A1*log
    s = torch.sum(log, axis=-1)
    return s


def pairwise_bc_similarity_gpu(A, B):
    A1 = A[:, None, :]
    A2 = B[None, :, :]
    mul = A1*A2
    mul = torch.sqrt(mul)
    s = torch.sum(mul, axis=-1)
    s = -torch.log(1 - s + 1e-6)
    return s


def s3vaada(net, unlabeled_loader, device, args, cycle, source_dataloader, new_dataloader):
    print("S3VAADA Sampling")

    print("alpha = ", args.alpha)
    print("beta = ", args.beta)

    vat = get_vat(net, unlabeled_loader, device, args)
    vat = (vat - vat.min())/(vat.max() - vat.min())
    softmax_output = get_softmax_output(net, unlabeled_loader, device, args)

    softmax_output = torch.Tensor(softmax_output).to(device)
    D = np.zeros((softmax_output.shape[0], softmax_output.shape[0]))
    b = 1000
    for i in range(0, softmax_output.shape[0], b):
        s1 = i
        e1 = min(i+b, softmax_output.shape[0])
        for j in range(0, softmax_output.shape[0], b):
            s2 = j
            e2 = min(j+b, softmax_output.shape[0])
            D[s1:e1, s2:e2] = pairwise_kl_gpu(
                A=softmax_output[s1:e1], B=softmax_output[s2:e2]).cpu().numpy()
    dists = D
    dists = (dists - dists.min())/(dists.max() - dists.min())

    similarity = np.zeros((softmax_output.shape[0], softmax_output.shape[0]))
    b = 1000
    for i in range(0, softmax_output.shape[0], b):
        s1 = i
        e1 = min(i+b, softmax_output.shape[0])
        for j in range(0, softmax_output.shape[0], b):
            s2 = j
            e2 = min(j+b, softmax_output.shape[0])
            similarity[s1:e1, s2:e2] = pairwise_bc_similarity_gpu(
                A=softmax_output[s1:e1], B=softmax_output[s2:e2]).cpu().numpy()
    similarity = (similarity - similarity.min()) / \
        (similarity.max() - similarity.min())

    number_of_unlabeled_samples = len(unlabeled_loader.dataset)
    S = []

    for i in range(args.budget):
        A = [j for j in range(number_of_unlabeled_samples) if j not in S]
        G = Gain(vat, dists, similarity, A, S,
                 alpha=args.alpha, beta=args.beta)
        S.append(A[G.argmax()])

    print(S)
    return torch.from_numpy(np.array(S))
