import torch

def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    - optimizer: optimizer for updating parameters
    - p: a variable for adjusting learning rate
    return: optimizer
    """
    for param_group in optimizer.param_groups:
        if "i_lr" not in param_group:
            param_group["i_lr"] = param_group["lr"]
        param_group['lr'] = param_group["i_lr"] / (1. + 10 * p) ** 0.75

    return optimizer

def sigmoid(parameter, plasticity):
    return 1. / (1 + torch.exp((-plasticity * parameter)))
