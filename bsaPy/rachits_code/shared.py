import torch
from datasets import generate_gg_blocks
import pdb  # noqa: F401
from utils import visualize_features


def compute_init(init_method, dataset, features):
    if init_method == 'truth':
        true_features = torch.from_numpy(generate_gg_blocks())
        extra_padding = torch.zeros(features - true_features.size(0), true_features.size(1)).double()
        return torch.cat([true_features, extra_padding], 0)
    elif init_method == 'nmf':
        try:
            from sklearn.decomposition import NMF
        except ImportError as e:
            print("sklearn required to compute the NMF solution. Please install sklearn (usually pip install sklearn)")
            raise e
        model = NMF(n_components=features)
        offset_dataset = dataset - dataset.min() + 1e-4
        model.fit_transform(offset_dataset)
        init = model.components_
        # reorder the NMF solution to make sense
        init = init[init.var(1).argsort()[::-1]]
        # fix the scaling on the solutions
        init = init/init.max(1).reshape((6, 1))
        visualize_features(init)
        return torch.from_numpy(init)
    elif init_method == 'random':
        return torch.zeros(features, dataset.shape[1]).uniform_(-0.5, 0.5)
    else:
        return None


class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad):
        return grad.clone()


class STRelaxedBernoulli(torch.distributions.RelaxedBernoulli):
    def rsample(self, *args, **kwargs):
        sample = super(STRelaxedBernoulli, self).rsample(*args, **kwargs)
        return Round.apply(sample)
