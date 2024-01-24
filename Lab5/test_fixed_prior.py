import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, plot_pred_test, finn_eval_seq, pred_test, show_result

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--model_path', type=str, help='path of the loaded model')
    parser.add_argument('--model_name', type=str, help='name of the loaded model')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=0, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=128, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=64, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=False, action='store_true')  

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    
    saved_model = torch.load('./%s/%s.pth' % (args.model_path, args.model_name))
    
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']

    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()

    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True)
    test_iterator = iter(test_loader)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    
    

    psnr_list = []
    for _ in tqdm(range(len(test_data) // args.batch_size)):  # a / b = c ... d, a // b = c
        # test
        try:
            test_seq, test_cond = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_loader)
            test_seq, test_cond = next(test_iterator)
        test_seq = test_seq.permute(1, 0, 2, 3 ,4).to(device)
        test_cond = test_cond.permute(1, 0, 2).to(device)
        pred_seq = pred_test(test_seq, test_cond, modules, args, device)
        # print(len(validate_seq[args.n_past:]))
        # print(len(pred_seq[args.n_past:]))
        _, _, psnr = finn_eval_seq(test_seq[args.n_past:], pred_seq[args.n_past:])
        psnr_list.append(psnr)
        
    ave_psnr = np.mean(np.concatenate(psnr_list))


    try:
        test_seq, test_cond = next(test_iterator)
    except StopIteration:
        test_iterator = iter(test_loader)
        test_seq, test_cond = next(test_iterator)
    test_seq = test_seq.permute(1, 0, 2, 3 ,4).to(device)
    test_cond = test_cond.permute(1, 0, 2).to(device)
    plot_pred_test(test_seq, test_cond, modules, args, device)
    
    print(f'Average PSNR: {ave_psnr}')

if __name__ == '__main__':
    main()
        