from __future__ import print_function
import sys
import numpy as np
import pdb  # noqa: F401
import argparse

import torch
import torch.utils.data
import torch.nn.init
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from datasets import GGBlocks
from utils import save_checkpoint, gen_id, input  # noqa: F401
from shared import compute_init
from ibp import MF_IBP_VAE

parser = argparse.ArgumentParser(description='VAEs for IBP and Paintbox models')

#
# Experiment parameters
#
parser.add_argument('--dataset', type=str, default='ggblocks',
                    help='dataset to train on')
parser.add_argument('--model', type=str, default='ibp',
                    help='which model to use (ibp, paintbox) - more will be added later')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--sigma-n', type=float, default=0.5,
                    help='noise stdev')
parser.add_argument('--dataset-size', type=int, default=500,
                    help='size of the training dataset')
parser.add_argument('--init', type=str, default=None,
                    help='method for initializing the mean of the dictionaries (A matrices): [random, truth, nmf]')

# Optimization parameters
parser.add_argument('--scheduler', type=int, default=None,
                    help='whether to decay the learning rate, and how often (i.e. epochs before multiplying the lr by 0.7)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--lr', type=float, default=3e-4,
                    help='learning rate')
parser.add_argument('--train-iwae', action='store_true',
                    help='use the IWAE loss as an optimization objective')
parser.add_argument('--n-samples', type=int, default=1,
                    help='number of samples for calculating IWAE loss. If ELBO is used instead for training, this'
                         'corresponds to the number of samples z ~ q(z) taken to evaluate the ELBO'
                         'NOTE: currently not implemented')

# Model parameters
parser.add_argument('--truncation', type=int, default=6,
                    help='number of sticks')
parser.add_argument('--alpha0', type=float, default=5.,
                    help='prior alpha for stick breaking Betas')
parser.add_argument('--hidden', type=int, default=500, help='hidden states')
parser.add_argument('--uuid', type=str, default=gen_id(), help='(somewhat) unique identifier for the model/job')
parser.add_argument('--temp', type=float, default=1.0, help='Temperature of Concrete approximation used')

# Logging + saving parameters
parser.add_argument('--train-from', type=str, default=None, metavar='M',
                    help='model to train from, if any')
parser.add_argument('--save', type=str, required=True,
                    help='where to save all of the logged output to [required]')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--traj', action='store_true',
                    help='whether to store per-epoch dumps of the trajectories')
parser.add_argument('--quiet', action='store_true',
                    help='whether to suppress the in-epoch prints')

args = parser.parse_args()

# Seed the experiment
torch.manual_seed(args.seed)

# Calculate device settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
cpu = torch.device('cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

if not args.quiet:
    print('running:\n' + ' '.join(sys.argv))


def main():
    if args.dataset == 'ggblocks':
        train_data = GGBlocks(N=args.dataset_size, sigma_n=args.sigma_n)
        test_data = GGBlocks(N=200, sigma_n=args.sigma_n)
    elif args.dataset == 'correlated':
        patterns = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1],
                             [1, 1, 0, 0],
                             [0, 0, 1, 1]])
        freq = np.array([0.1, 0.1, 0.1, 0.1, 0.3, 0.3])
        train_data = GGBlocks(N=args.dataset_size, sigma_n=args.sigma_n, patterns=patterns, freq=freq)
        test_data = GGBlocks(N=200, sigma_n=args.sigma_n, patterns=patterns, freq=freq)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, **kwargs)

    # compute the initialization of the model
    init = compute_init(args.init, train_data.train_data, args.truncation)

    # cross-model dictionary to initialize it
    model_kwargs = {
        'max_truncation_level': args.truncation,
        'alpha0': args.alpha0,
        'sigma_n': args.sigma_n,
        'init': init
    }

    # All switching logic is here.
    if args.model == 'ibp':
        model_cls = MF_IBP_VAE
    else:
        raise NotImplementedError

    # set up objects for use
    model = model_cls(**model_kwargs).to(device)
    model_kwargs.pop('init')
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr)
    if args.scheduler is not None:
        scheduler = StepLR(optimizer, args.scheduler, 0.7)

    logs = []

    def flush(params):
        if args.traj:
            params.update({'model': model.to(cpu)})
        logs.append(params)

    # run experiment
    try:
        best_score = np.inf
        for epoch in range(1, args.epochs + 1):
            train_score = model.train_epoch(train_loader, optimizer, epoch, args, device, args.n_samples)
            eval_score = model.evaluate(test_loader, args, device, 10)
            if eval_score['ELBO'] < best_score:
                model_to_save = model_cls(**model_kwargs).to(cpu)
                model_to_save.load_state_dict(model.state_dict())

            if not args.quiet:
                s = ''.join(["| {} {:<5.1f}".format(k, v) for k, v in eval_score.items()])
                s += '| Train: {:.1f}'.format(train_score)
                print("[Epoch {:<4}] ".format(epoch) + s)

            # print(model.A_mean.sum().item())
            flush(eval_score)
            if args.scheduler is not None:
                scheduler.step()
    except KeyboardInterrupt:
        save = input("\n save learned features and trajectories anyway? [y/n] ")
        if save.strip() not in ('y', 'yes'):
            sys.exit(0)

    # Save everything here
    torch.save({'model': model_to_save, 'logs': logs}, args.save)

    # Just for debugging, save the output learned features to `learned_features.npy`
    np.save('learned_features.npy', model.A_mean.data.to(cpu).numpy())


if __name__ == '__main__':
    main()
