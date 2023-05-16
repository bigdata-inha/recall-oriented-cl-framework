import torch
from operator import mul
from functools import reduce
import numpy as np

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList


def LogSumExp(x, dim=0):
    m, _ = torch.max(x, dim=dim, keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim, keepdim=True))

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(model, trainloader, device):   # verbose=False, subset=None, **kwargs
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0

    with torch.no_grad():
        # if subset is not None:
        #     num_batches = int(num_batches * subset)
        #     loader = itertools.islice(loader, num_batches)
        # if verbose:
        #     loader = tqdm.tqdm(loader, total=num_batches)
        for batch_idx, (inputs, _) in enumerate(trainloader):
            inputs = inputs.to(device)
            input_var = torch.autograd.Variable(inputs)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def make_chunk_size(n_param, n_condition=[100, 100, 100]):
    n_param = torch.tensor(n_param)
    n_condition = torch.tensor(n_condition)

    chunk_size = n_param // n_condition + 1
    remainder = chunk_size * n_condition - n_param

    return chunk_size, remainder


def param_insert(new_model, fake_param):
    resized_chunks = []
    shape = [param.data.shape for param in new_model.parameters()]
    chunk_sizes = [reduce(mul, dims, 1) for dims in shape]
    param_chunk = torch.split(fake_param.view(-1), split_size_or_sections=chunk_sizes)

    for index, param in enumerate(param_chunk):
        param = torch.reshape(param, shape[index])
        resized_chunks.append(param)

    for idx, param in enumerate(new_model.parameters()):
        param.data = resized_chunks[idx]

    return new_model

