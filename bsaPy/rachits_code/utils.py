# utils for hacking around
import torch
from torch.autograd import Variable
import math
import sys
import numpy as np
import shutil
import pdb
import uuid


def show_memusage(device=1):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))


SMALL = 1e-16
EULER_GAMMA = 0.5772156649015329

try:
    input = raw_input
except NameError:
    input = input


def binarize(image):
    return image.bernoulli()


def save_dataset(state, filename):
    name = '{}_data.pth.tar'.format(filename)
    torch.save(state, name)


def save_checkpoint(state, is_best, filename='checkpoint'):
    name = '{}.pth.tar'.format(filename)
    torch.save(state, name)
    if is_best:
        best_name = '{}_best.pth.tar'.format(filename)
        shutil.copyfile(name, best_name)


def print_grad(name):
    def hook(grad):
        if math.isnan(grad.sum().data[0]):
            print("grad failed for: {}".format(name))
            pdb.set_trace()
        # print("dL/d{}: {}".format(name, grad.sum()))
    return hook


def gen_id():
    return str(uuid.uuid4()).replace("-", "")


def isnan(v):
    return np.isnan(v.cpu().numpy()).any()


def findnan(v):
    if v.__class__ == Variable:
        v = v.data
    return np.isnan(v.cpu().numpy())


def log_sum_exp(tensor, dim=-1, keepdim=False):
    """
    Numerically stable implementation for the `log-sum-exp` operation. The
    summing is done along `dim` axis
    Args:
        tensor (torch.Tensor)
        dim (int): Which axis to apply the LSE over
        keepdim (Boolean): Whether to retain the dimension on summing.
    """
    max_val = tensor.max(dim=dim, keepdim=True)[0]
    tmp = (tensor - max_val).exp().sum(dim=dim, keepdim=keepdim).log()
    if not keepdim:
        max_val = max_val.squeeze(dim)
    return max_val + tmp


def visualize_features(arr, name=''):
    if type(arr) is torch.Tensor:
        arr = arr.cpu().numpy()
    from matplotlib import pyplot as plt
    f, axes = plt.subplots(3, 2)
    a, b = arr.min(), arr.max()
    print("min: {:.2f}, max: {:.2f}".format(a, b))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(arr[i].reshape((6, 6)), cmap='Greys', vmin=a, vmax=b, interpolation=None)
        ax.set_title(i)
    f.suptitle(name)
    f.tight_layout()
    f.subplots_adjust(top=0.88)
    plt.show()


def visualize_before_after(before, after, name=''):
    if type(before) is torch.Tensor:
        before = before.cpu().numpy()
    if type(after) is torch.Tensor:
        after = after.cpu().numpy()
    from matplotlib import pyplot as plt
    f, axes = plt.subplots(3, 4)
    a, b = min(before.min(), after.min()), max(before.max(), after.max())
    print("min: {:.2f}, max: {:.2f}".format(a, b))
    for i in range(3):
        for j in range(4):
            ax = axes[i][j]
            if j < 2:
                ax.imshow(before[i * 2 + j].reshape((6, 6)), cmap='Greys', vmin=a, vmax=b, interpolation=None)
            else:
                ax.imshow(after[i * 2 + (j - 2)].reshape((6, 6)), cmap='Greys', vmin=a, vmax=b, interpolation=None)
    f.suptitle(name)
    f.tight_layout()
    f.subplots_adjust(top=0.88)
    plt.show()


# crucial
def logit(x):
    return (x + SMALL).log() - (-x + SMALL).log1p()


def print_in_epoch_summary(epoch, batch_idx, batch_size, dataset_size, loss, NLL, KLs):
    kl_string = '\t'.join(['KL({}): {:.3f}'.format(key, val / batch_size) for key, val in KLs.items()])
    print('Train Epoch: {} [{:<5}/{} ({:<2.0f}%)]\tLoss: {:.3f}\tNLL: {:.3f}\t{}'.format(
        epoch, (batch_idx + 1) * batch_size, dataset_size,
        100. * (batch_idx + 1) / (dataset_size / batch_size),
        loss / batch_size,
        NLL / batch_size,
        kl_string))
    sys.stdout.flush()


def print_epoch_summary(epoch, loss):
    print('====> Epoch: {:<3} Average loss: {:.4f}'.format(epoch, loss))
    sys.stdout.flush()
