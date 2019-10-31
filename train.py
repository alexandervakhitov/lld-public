import torch
import data.batched as ba
import cnn.net_multibatch as nmb
import torch.optim as optim
import os
import train.multibatch_trainer as mbt
import tqdm
from torch.autograd import Variable
import numpy as np

dir_path = '../traindata/'
ptnum = 5

is_noisy = False
def get_net():
    return nmb.FeatureEncoder(is_cuda=True, is_color=False, is_pyramid = False, depth=3, g=0).cuda()

def train_multibatch(se=-1):
    train_loader, test_loader, test_loader_2 = ba.get_combined_training_v2(pt_per_line=ptnum, is_noisy=is_noisy)
    encoder_net = get_net()
    optimizer = optim.Adam(encoder_net.parameters(), lr = 1e-4)
    mbt.run_training(dir_path, train_loader, test_loader, encoder_net, optimizer, is_triplet=False, start_epoch=se, vdseqid='7')

def eval_multibatch(ep_id):
    train_loader, test_loader, test_loader2  = ba.get_combined_training(pt_per_line=ptnum)
    encoder_net = get_net()
    mbt.run_validation(dir_path, test_loader2, encoder_net, ep_id, is_triplet=False, vqseqid='9', is_save=True)


def test_with_descriptors_hetero(ep_id=0):
    encoder_net = get_net()

    kitti_ids = [8, 9, 10]
    seq_ids_kitti = np.arange(14, 17)

    euroc_ids = [2, 4, 6]
    seq_ids_euroc = np.arange(17, 20)

    seq_map = {}
    for i in range(0, len(kitti_ids)):
        seq_map[seq_ids_kitti[i]] = ('kitti', kitti_ids[i])

    for i in range(0, len(euroc_ids)):
        seq_map[seq_ids_euroc[i]] = ('euroc', euroc_ids[i])

    mbt.run_test_heterogen(dir_path, encoder_net, seq_map, ep_id=ep_id, is_triplet=False, do_savedesc=True,
                           pt_per_line=ptnum)

train_multibatch()

eval_multibatch(1)

test_with_descriptors_hetero(1)
