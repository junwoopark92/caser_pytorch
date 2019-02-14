import sys
sys.path.extend(['..', '.'])
from time import time
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch.autograd import Variable
from caser import Caser
from evaluation import evaluate_ranking
from collections import defaultdict
from interactions import Interactions
from utils import *

import pickle
import argparse
import pandas as pd
from train_caser import Recommender


# d=100,nv=2,nh=16,drop=0.5,ac_conv=iden,ac_fc=sigm for Gowalla data
# d=50,nv=4,nh=16,drop=0.5,ac_conv=relu,ac_fc=relu for MovieLens data
# d=50, nv=4,nh=16,drop=0.5,ac_conv=relu,ac_fc=relu (default)

params = {
    'L': [1,2,3,4,5,6,7,8,9],
    'T': [1,2,3],
    'nv':[2],
    'nh':[16],
    'drop': [0.5],
    'd': [100],
    'ac_conv':['iden'],
    'ac_fc':['sigm'],
    'batch_size': [512]
}


def combination_param(**kwargs):
    res = [{}]
    for key, vals in kwargs.items():
        res = [dict(x, **{key: y}) for x in res for y in vals]
    return res


def param2arglist(param):
    return ' '.join(["--{} {}".format(key, value) for key, value in param.items()]).split(' ')


def train(config):
    # set seed
    set_seed(config.seed,
             cuda=config.use_cuda)

    # load dataset
    train = Interactions(config.train_root)
    # transform triplets to sequence representation
    train.to_sequence(config.L, config.T)

    test = Interactions(config.test_root,
                        user_map=train.user_map,
                        item_map=train.item_map)

    # print(config)
    # print(model_config)
    # fit model
    model = Recommender(n_iter=config.n_iter,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate,
                        l2=config.l2,
                        neg_samples=config.neg_samples,
                        model_args=model_config,
                        use_cuda=config.use_cuda)

    return model.fit(train, test, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', type=str, default='datasets/gowalla/test/train.txt')
    parser.add_argument('--test_root', type=str, default='datasets/gowalla/test/test.txt')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    parser.add_argument('--n_iter', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=str2bool, default=False)
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--nv', type=int, default=2)
    parser.add_argument('--nh', type=int, default=16)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--ac_conv', type=str, default='relu')
    parser.add_argument('--ac_fc', type=str, default='sigm')

    param_list = combination_param(**params)

    results = []
    print(param_list)
    for param in param_list:
        arglist = param2arglist(param)
        print(arglist)
        model_config = parser.parse_args(arglist)
        output_str, maps = train(model_config)
        results.append((' '.join(arglist), output_str, maps))
        print(' '.join(arglist), output_str, maps)

    result_df = pd.DataFrame(results, columns=['args', 'metrics', 'map'])
    result_df = result_df.sort_values(by=['map'])
    result_df.to_csv('./tune_result.csv', index=False)









