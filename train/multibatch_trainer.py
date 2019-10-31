import torch
import os
import time
import numpy as np
import data.batched as ba
import torch.optim as optim
import cnn.net_multibatch as nmb

def compose_batch(batch):
    batch = batch[0]
    n = len(batch[0])
    ims = np.asarray(batch[0]).astype(float)
    ims = torch.from_numpy(ims).float()
    lines = batch[1]
    ln_lens = [l.shape[0] for l in lines]
    ln_max = np.max(np.asarray(ln_lens))
    pt_num = lines[0].shape[2]
    lines_torch = torch.zeros(n, ln_max, pt_num, 2)
    for i in range(0, n):
        lines_torch[i, 0:lines[i].shape[0], :, :] = torch.from_numpy(lines[i].astype(float)).float()

    negs = batch[2]
    poss = batch[3]
    n_pos = len(poss[0])

    max_neg = max([max([len(neg_for_pos) for neg_for_pos in neg]) for neg in negs])
    neg_t = -1*torch.ones(n, n_pos, max_neg)
    for i in range(0, len(negs)):
        for j in range(0, len(negs[i])):
            neg_for_pos = negs[i][j]
            if len(neg_for_pos)>0:
                neg_t[i][j][0:len(neg_for_pos)] = torch.from_numpy(np.asarray(neg_for_pos).astype(long))

    pos_t = -1*torch.ones(n, n_pos)
    for i in range(0, len(poss)):
        p = poss[i]
        for j in range(0, len(p)):
            if len(p[j]) == 1:
                pos_t[i, j] = p[j][0]

    if len(batch) == 4:
        return ims, lines_torch, neg_t, poss  # , batch[0][4], batch[0][5]
    else:
        return ims, lines_torch, neg_t, poss, batch[4], batch[5], batch[6]

def run_training(dir_path, train_loader, test_loader, encoder_net, optimizer, start_epoch=-1, is_triplet = True, vdseqid='7'):

    if start_epoch >= 0:
        checkpoint = torch.load(dir_path + '/' + str(start_epoch) + '.pyh.tar')
        encoder_net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print 'loaded epoch ' + str(start_epoch)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for ep_id in range(start_epoch + 1, 8):
        # nmb.net_test(train_loader, encoder_net)
        encoder_net = nmb.net_train(train_loader, encoder_net, optimizer, is_triplet=is_triplet)
        valdesc_folder = dir_path+"/val_descs_"+str(ep_id)+"/"+vdseqid+'/'
        if not os.path.exists(valdesc_folder):
            os.makedirs(valdesc_folder)
        nmb.net_test(test_loader, encoder_net, save_descs=True, save_folder=valdesc_folder)
        torch.save({
            'epoch': ep_id,
            'state_dict': encoder_net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, dir_path + '/' + str(ep_id) + '.pyh.tar')

def run_validation(dir_path, test_loader, encoder_net, ep_id, is_triplet = True, is_save=False, vqseqid=''):
    # for ep_id in range(start_epoch + 1, 10):
    checkpoint = torch.load(dir_path + '/' + str(ep_id) + '.pyh.tar')
    encoder_net.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    print 'loaded epoch ' + str(ep_id)
    val_descs_save_path = dir_path+'/val_descs_'+str(ep_id)+'/'+vqseqid+'/'
    if not os.path.exists(val_descs_save_path):
        os.makedirs(val_descs_save_path)
    curr_ap = nmb.net_test(test_loader, encoder_net, is_triplet=is_triplet, save_descs=is_save, save_folder=val_descs_save_path )
    rep_txt = open(dir_path + '/report.txt', 'a')
    rep_txt.write(str(ep_id) + ':' + str(curr_ap) + '\n')


def run_test_heterogen(dir_path, encoder_net, seq_map, ep_id=0, is_triplet = True, do_savedesc=True, pt_per_line=5, n_lim=6):

    checkpoint = torch.load(dir_path + '/' + str(ep_id) + '.pyh.tar')
    encoder_net.load_state_dict(checkpoint['state_dict'])
    print 'loaded epoch ' + str(ep_id)
    for seq_id in seq_map:
        seq_type, seq_save_code = seq_map[seq_id]
        test_loader = ba.get_combined_test(seq_id, pt_per_line, n_lim)
        if seq_type == 'kitti':
            save_dir = dir_path + '/descs' + '/' + str(seq_save_code) + '/'
        else:
            save_dir = dir_path + '/descs_euroc' + '/' + str(seq_save_code) + '/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        nmb.net_test(test_loader, encoder_net, is_triplet = is_triplet, save_descs=do_savedesc, save_folder = save_dir)



def run_inference(dir_path, encoder_net, ep_id=0, do_savedesc=True):
    checkpoint = torch.load(dir_path + '/' + str(ep_id) + '.pyh.tar')
    encoder_net.load_state_dict(checkpoint['state_dict'])
    print 'loaded epoch ' + str(ep_id)
    inf_loader = ba.get_loader()
    nmb.net_inference(inf_loader, encoder_net, save_descs=do_savedesc, save_folder=dir_path+'/inf_descs')
