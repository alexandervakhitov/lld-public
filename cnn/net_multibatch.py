import torch.nn as nn
import torch
import torch.nn.functional as F
import tqdm
from torch.autograd import Variable
import torch.optim as optim
import time
import sklearn.metrics as metrics
import numpy as np

def compute_distances(x, pos_inds, neg_mask):
    #x: b x C x N
    #pos_mask: b x n_p, pos_inds: b x n_p, neg_mask b x N x np
    pos_mask = (pos_inds >= 0).float()
    pos_anti_mask = (pos_inds<0)
    b, np = pos_inds.shape
    C = x.shape[1]
    pos_inds[pos_anti_mask] = 0
    pos_inds_exp = pos_inds.view(b, 1, np).expand(-1, C, -1)
    # x_pos = torch.index_select(x, 2, pos_inds_exp)
    x_pos = torch.gather(x, 2, pos_inds_exp)
    #x_pos b x C x n_p
    x_pos1 = x_pos[1:, :, :].permute(0, 2, 1).contiguous().view(-1, C)
    x_pos0 = x_pos[0]
    x_pos_from = x_pos0.view(1, C, np).expand(b-1, -1, -1).permute(0, 2, 1).contiguous().view(-1, C)
    # pos_mask_flat = pos_mask[1:,:].reshape(-1)
    pos_mask_part = pos_mask[1:,:]
    dists_pos = F.pairwise_distance(x_pos_from, x_pos1).view(b-1, np)

    # d_pos = torch.sum(F.pairwise_distance(x_pos_from, x_pos1) * pos_mask_flat)
    N = x.shape[2]
    x_pos_from = x_pos0.view(1, 1, C, np).expand(b-1, N, -1, -1).permute(0, 1, 3, 2).contiguous().view(-1, C)
    x_to = x[1:].permute(0, 2, 1).view(b-1, N, 1, C).expand(-1, -1, np, -1).contiguous().view(-1, C)
    # print x_pos_from.shape
    # print x_to.shape
    # print b
    # print N
    dists_flat = F.pairwise_distance(x_pos_from, x_to)
    dists = dists_flat.view(b-1, N, np)

    neg_mask_part = neg_mask[1:].float()
    max_dist = 2.0
    dists_neg = dists * neg_mask_part + max_dist * (torch.ones_like(neg_mask_part) - neg_mask_part)

    #dists_pos: b-1 x np
    #pos_mask_part: b-1 x np
    #dists_neg: b-1 x N x np
    #neg_mask_part: b-1 x N x np
    return dists_pos, pos_mask_part, dists_neg, neg_mask_part

def compute_trip_loss(x, pos_inds, neg_mask, m):
    dists_pos, pos_mask_part, dists_neg, neg_mask_part = compute_distances(x, pos_inds, neg_mask)
    # d_pos_flat = dists_pos.view(-1)[(pos_mask_part>0).view(-1)].detach().cpu().numpy()
    # print('+ dists '+str(np.min(d_pos_flat))+' '+str(np.max(d_pos_flat)))

    d_pos = torch.sum(dists_pos * pos_mask_part, 0)

    # d_neg_flat = dists_neg.view(-1)[(neg_mask_part>0).view(-1)].detach().cpu().numpy()
    # print('- dists ' + str(np.min(d_neg_flat)) + ' ' + str(np.max(d_neg_flat)))

    # print(dists_neg.shape)
    d1, a1 = torch.min(dists_neg, dim=0)
    # print (d1.shape)
    d_neg, a_neg = torch.min(d1, 0)
    # print(d_neg.shape)
    # m_var = torch.ones(1).cuda() * m
    d_neg_diff = m - d_neg
    # d_nd_np = d_neg_diff.detach().cpu().numpy()
    # print('min neg dists '+str(np.min(d_nd_np))+' : '+str(np.max(d_nd_np)))
    # print(d_neg_diff.shape)
    mask = (d_neg_diff>=0).float()
    # print((mask * d_neg_diff).shape)
    # print(d_pos.shape)
    return torch.sum(mask * d_neg_diff + d_pos)

def compute_hn_loss(x, pos_inds, neg_mask, m):
    dists_pos, pos_mask_part, dists_neg, neg_mask_part = compute_distances(x, pos_inds, neg_mask)
    # d_pos_flat = dists_pos.view(-1)[(pos_mask_part>0).view(-1)].detach().cpu().numpy()
    # print('+ dists '+str(np.min(d_pos_flat))+' '+str(np.max(d_pos_flat)))

    d_pos = torch.sum(dists_pos * pos_mask_part, 0)

    # d_neg_flat = dists_neg.view(-1)[(neg_mask_part>0).view(-1)].detach().cpu().numpy()
    # print('- dists ' + str(np.min(d_neg_flat)) + ' ' + str(np.max(d_neg_flat)))

    # print(dists_neg.shape)
    d1, a1 = torch.min(dists_neg, dim=0)
    # print (d1.shape)
    d_neg, a_neg = torch.min(d1, 0)
    d_diff = d_pos - d_neg
    # print(d_neg.shape)
    # m_var = torch.ones(1).cuda() * m
    d_shifted = d_diff + m
    # d_nd_np = d_neg_diff.detach().cpu().numpy()
    # print('min neg dists '+str(np.min(d_nd_np))+' : '+str(np.max(d_nd_np)))
    # print(d_neg_diff.shape)
    mask = (d_shifted>=0).float()
    # print((mask * d_neg_diff).shape)
    # print(d_pos.shape)
    return torch.sum(mask * d_shifted)



def compute_distvecs(x, pos_inds, neg_mask):
    dists_pos, pos_mask_part, dists_neg, neg_mask_part = compute_distances(x, pos_inds, neg_mask)
    d_pos = np.zeros(0)
    d_neg = np.zeros(0)
    pos_mask_flat = pos_mask_part.view(-1)
    dists_pos_flat = dists_pos.view(-1)
    if pos_mask_flat.shape[0] > 0:
        pos_inds = pos_mask_flat>0
        d_pos = dists_pos_flat.view(-1)
        if dists_pos_flat.shape[0]>0:
            d_pos = d_pos[pos_inds]
            if d_pos.shape[0]>0:
                d_pos = d_pos.detach().cpu().numpy()

    neg_mask_flat = dists_neg.view(-1)
    if neg_mask_flat.shape[0] > 0:
        neg_inds = neg_mask_flat<2.0
        dists_neg_flat = dists_neg.view(-1)
        if dists_neg_flat.shape[0]>0:
            d_neg_t = dists_neg_flat[neg_inds]#.detach().cpu().numpy()
            if d_neg_t.shape[0]>0:
                d_neg = d_neg_t.detach().cpu().numpy()
    return d_pos, d_neg



def sample_descriptors(x, lines, w_img, h_img):
    w_map = x.shape[3]
    h_map = x.shape[2]
    # print(x.shape)
    m_x = (w_img-w_map)/2.0
    m_y = (h_img-h_map)/2.0
    # print(w_img)
    # print(m_x)
    lines_x_flat = lines[:, :, :, 0].view(-1)
    good_lines = (lines_x_flat > 0).nonzero()
    # print(str(len(good_lines))+ ' ' + str(lines_x_flat.shape[0]))
    lines_x_flat[good_lines] = lines_x_flat[good_lines] + m_x*torch.ones_like(lines_x_flat[good_lines])
    lines_x_flat[good_lines] = 2.0/w_map*lines_x_flat[good_lines] - torch.ones_like(lines_x_flat[good_lines])
    lines[:, :, :, 0] = lines_x_flat.view(lines[:, :, :, 0].shape)

    lines_y_flat = lines[:, :, :, 1].view(-1)
    good_lines = (lines_y_flat > 0).nonzero()
    lines_y_flat[good_lines] = lines_y_flat[good_lines] + m_y*torch.ones_like(lines_y_flat[good_lines])
    lines_y_flat[good_lines] = 2.0 / h_map * lines_y_flat[good_lines] - torch.ones_like(lines_y_flat[good_lines])
    lines[:, :, :, 1] = lines_y_flat.view(lines[:, :, :, 1].shape)

    lds = F.grid_sample(x, lines, mode='bilinear', padding_mode='border')
    avg_lds = torch.sum(lds, 3)
    avg_lds = F.normalize(avg_lds, p=2, dim=1)
    return avg_lds

class FeatureEncoder(nn.Module):

    def initialize_l2(self, g=0):
        if g==0:
            g = 4
            if self.depth == 2:
                g = 2
            if self.depth == 3:
                g = 1
        print 'initializing depth= '+str(self.depth)+ 'g='+str(g)
        #        g = 1

        self.downsample_init = nn.Upsample(scale_factor=0.5, mode='bilinear')

        if self.is_color:
            self.conv1 = nn.Conv2d(3, 8 * g, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(1, 8 * g, kernel_size=3, stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(8 * g)
        self.conv2 = nn.Conv2d(8 * g, 8 * g, kernel_size=3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(8 * g)
        if self.depth > 0:
            self.conv3 = nn.Conv2d(8 * g, 16 * g, kernel_size=3, stride=2, padding=1)  # 1/2
        else:
            self.conv3 = nn.Conv2d(8 * g, 16 * g, kernel_size=3, stride=1, padding=1)  # 1/2
        self.batch3 = nn.BatchNorm2d(16 * g)
        self.conv4 = nn.Conv2d(16 * g, 16 * g, kernel_size=3, stride=1, padding=1)
        self.batch4 = nn.BatchNorm2d(16 * g)
        k = 16 * g
        if self.depth >= 2:
            self.conv5 = nn.Conv2d(16 * g, 32 * g, kernel_size=3, stride=2, padding=1)  # 1/4
            self.batch5 = nn.BatchNorm2d(32 * g)
            self.conv6 = nn.Conv2d(32 * g, 32 * g, kernel_size=3, padding=1)
            self.batch6 = nn.BatchNorm2d(32 * g)
            k = 32*g
        if self.depth == 3:
            self.conv61 = nn.Conv2d(32 * g, 64 * g, kernel_size=3, stride=2, padding=1)  # 1/4
            self.batch61 = nn.BatchNorm2d(64 * g)
            self.conv62 = nn.Conv2d(64 * g, 64 * g, kernel_size=3, padding=1)
            self.batch62 = nn.BatchNorm2d(64 * g)
            k = 64*g
            if self.is_skip:
                self.convu1 = nn.Conv2d(64 * g, 64 * g, kernel_size=3, padding=1)  #
                self.batchu1 = nn.BatchNorm2d(64 * g)
                #  1/4
                self.convu2 = nn.Conv2d(64 * g, 64 * g, kernel_size=3, padding=1)  # 1/4
                self.batchu2 = nn.BatchNorm2d(64 * g)
                self.convu3 = nn.Conv2d(64 * g, 64 * g, kernel_size=3, padding=1)  # 1/4
                self.batchu3 = nn.BatchNorm2d(64 * g)

        # self.conv7 = nn.Conv2d(k, k, kernel_size=8, padding=0)
        self.conv7 = nn.Conv2d(k, k, kernel_size=7, padding=3)
        self.batch7 = nn.BatchNorm2d(k)

        self.dropout = nn.Dropout2d()

        if self.is_learnable:
            self.deconv_64 = nn.ConvTranspose2d(k, k, stride=8, kernel_size=3)
            self.deconv_32 = nn.ConvTranspose2d(k, k, stride=4, kernel_size=3)
            self.deconv_16 = nn.ConvTranspose2d(k, k, stride=2, kernel_size=3)
        else:
            self.deconv_64 = nn.Upsample(scale_factor=8, mode='bilinear')
            self.deconv_32 = nn.Upsample(scale_factor=4, mode='bilinear')
            self.deconv_16 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.deconv_final = nn.Upsample(scale_factor=self.upscale_factor, mode='bilinear')

    def __init__(self, is_cuda, is_color=False, upscale_factor=1, is_pyramid=True, depth=2, g=0,
                 is_learnable=False, is_skip = False):
        super(FeatureEncoder, self).__init__()
        self.margins = -1*torch.ones(2).float().cuda()
        self.final_size = -1 * torch.ones(2).float().cuda()
        self.is_learnable = is_learnable
        self.depth = depth
        self.upscale_factor = upscale_factor
        self.is_color = is_color
        self.is_skip = is_skip
        print 'my net init start full'
        self.initialize_l2(g)
        # self._initialize_weights()
        print 'init L2 done'

        self.is_cuda = is_cuda
        self.scale = 1.0
        # self.is_plain = is_plain

        self.is_pyramid = is_pyramid
        self.is_test_mode = False

    def forward(self, x):
        # print 'start '+str(x.shape)
        init_shape = x.shape
        x = F.relu(self.batch1(self.conv1(x)), inplace=True)
        # print 'b1 ' + str(x.shape)
        x = F.relu(self.batch2(self.conv2(x)), inplace=True)
        x2 = x
        # print 'b2 ' + str(x.shape)
        x = F.relu(self.batch3(self.conv3(x)), inplace=True)

        # print 'b3 ' + str(x.shape)
        x = F.relu(self.batch4(self.conv4(x)), inplace=True)
        if (self.depth == 0):
            if self.is_skip:
                w = x2.shape[3]
                h = x2.shape[2]

                xm_1 = x[:, 0:x2.shape[1], 0:h, 0:w] + x2
                xm_2 = x[:, x2.shape[1]:, :, :]
                xm = torch.stack([xm_1, xm_2], dim=1)
                xm = xm.view(x.shape)
                x = xm
                # x[:, 0:x2.shape[1], 0:h, 0:w] = x[:, 0:x2.shape[1], 0:h, 0:w] + x2
            x = self.conv7(x)
            return x
        # print 'b4 ' + str(x.shape)
        if (self.depth == 1):
            x = self.batch7(self.conv7(x))
            # print 'b7 ' + str(x.shape)
            x = self.deconv_16(x)
            if self.is_skip:
                w = x2.shape[3]
                h = x2.shape[2]
                x[:, 0:x2.shape[1], 0:h, 0:w] = x[:, 0:x2.shape[1], 0:h, 0:w] + x2

            x = self.deconv_final(x)
            return x

        x4 = x

        x = F.relu(self.batch5(self.conv5(x)), inplace=True)
        # print 'b5 ' + str(x.shape)
        x = F.relu(self.batch6(self.conv6(x)), inplace=True)
        # print 'b6 ' + str(x.shape)

        if self.depth == 2:
            x = self.batch7(self.conv7(x))
            # print 'b7 ' + str(x.shape)
            x = self.deconv_32(x)

            x = self.deconv_final(x)
            return x
        x8 = x
        x = self.batch61(self.conv61(x))
        x = self.batch62(self.conv62(x))
        x = self.batch7(self.conv7(x))
        # print 'b7 ' + str(x.shape)
        if self.is_skip:
            # print('before ups '+str(x.shape))
            x = F.upsample_bilinear(x, x8.shape[2:])
            # print('after ups ' + str(x.shape))
            x = x + x8.repeat(1, 2, 1, 1)
            x = self.batchu1(self.convu1(x))
            x = F.upsample_bilinear(x, x4.shape[2:])
            # print(x4.shape)
            x4r = x4.repeat(1, 4, 1, 1)
            # print(x4r.shape)
            x = x + x4r
            x = self.batchu2(self.convu2(x))
            x = F.upsample_bilinear(x, init_shape[2:])
            x = x + x2.repeat(1,8,1,1)
            x = self.batchu3(self.convu3(x))
        else:
            x = self.deconv_64(x)
            x = self.deconv_final(x)
        return x

def net_train(loader, encoder_net, optimizer, is_triplet=True):
    encoder_net.train()
    avg_loss = 0
    for data in tqdm.tqdm(loader):
        x = Variable(data['images'].cuda(), requires_grad=False)
        lines = Variable(data['lines'].cuda(), requires_grad=False)
        # torch.cuda.synchronize()
        # torch.cuda.synchronize()
        # t0 = time.time()
        y = encoder_net(x)
        # torch.cuda.synchronize()
        # torch.cuda.synchronize()
        # t1 = time.time()
        w_img = x[0].shape[2]
        h_img = x[0].shape[1]
        d = sample_descriptors(y, lines, w_img, h_img)
        # d_np = d.detach().cpu().numpy()
        # print(np.linalg.norm(d_np[0,:,0]))
        pos_inds = Variable(data['positives'].cuda(), requires_grad=False)
        neg_mask = Variable(data['negatives'].cuda(), requires_grad=False)
        l = 0
        if is_triplet:
            l = compute_trip_loss(d, pos_inds, neg_mask, m=0.5)
        else:
            l = compute_hn_loss(d, pos_inds, neg_mask, m=0.5)
        avg_loss += l.detach().cpu().numpy()
        optimizer.zero_grad()
        # print(l.detach().cpu().numpy())
        l.backward()
        # torch.cuda.synchronize()
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print 'fwd time '+str(t1 - t0)
        # print 'bwd time ' + str(t2 - t1)
        optimizer.step()

    print('avg train loss '+str(avg_loss/len(loader)))
    return encoder_net

def get_ap(all_pos, all_neg):
    all_pos = np.concatenate(tuple(all_pos), axis=0)
    y_true_pos = np.ones(all_pos.shape[0])
    all_neg = np.concatenate(tuple(all_neg), axis=0)
    y_true_neg = np.zeros(all_neg.shape[0])
    y_true = np.concatenate((y_true_pos, y_true_neg), axis=0)
    y_est = 1.0 - 0.5 * np.concatenate((all_pos, all_neg), axis=0)
    curr_ap = metrics.average_precision_score(y_true, y_est)
    return curr_ap

def net_test(loader, encoder_net, is_triplet = True, save_descs=False, save_folder = ''):
    encoder_net.eval()
    with torch.no_grad():
        all_pos = []
        all_neg = []
        cnt = 0
        avg_loss = 0
        avg_time = 0
        for data in tqdm.tqdm(loader):
            x = Variable(data['images'].cuda(), requires_grad=False)
            if len(x) == 0:
                continue
            # print('bs='+str(len(x)))
            lines = Variable(data['lines'].cuda(), requires_grad=False)
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t0 = time.time()
            y = encoder_net(x)
            w_img = x[0].shape[2]
            h_img = x[0].shape[1]
            d = sample_descriptors(y, lines, w_img, h_img)

            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t1 = time.time()
            # print(str(t1 - t0) + ' sec')
            l = 0
            pos_inds = Variable(data['positives'].cuda(), requires_grad=False)
            neg_mask = Variable(data['negatives'].cuda(), requires_grad=False)
            if is_triplet:
                l = compute_trip_loss(d, pos_inds, neg_mask, m=0.5)
            else:
                l = compute_hn_loss(d, pos_inds, neg_mask, m=0.5)
            l = l.detach().cpu().numpy()
            avg_loss += l
            avg_time += t1-t0


            # pos_inds = Variable(data['positives'].cuda(), requires_grad=False)
            # neg_mask = Variable(data['negatives'].cuda(), requires_grad=False)
            # d_pos, d_neg = compute_distvecs(d, pos_inds, neg_mask)

            # cnt += 1
            # all_pos.append(d_pos)
            # all_neg.append(d_neg)

            if save_descs:
                img_ids = data['image_ids']
                pair_ids = data['pair_ids']
                ln_lens = data['ln_lens']
                for ii in range(0, len(img_ids)):
                    img_id = img_ids[ii]
                    pair_id = pair_ids[ii]
                    f_out = open(save_folder + '/' + str(img_id) + '_' + str(pair_id) + '.txt', 'w')
                    lines_np = lines.detach().cpu().numpy()
                    descs_np = d.detach().cpu().numpy()
                    for lind in range(0, ln_lens[ii]):
                        # if lines_np[ii, lind, 0, 0] == 0 and lines_np[ii, lind, -1, 0] == 0:
                        #     # print('breaking the cycle iid ' +str(img_id))
                        #     # print(descs_np[ii, 0, lind:lind+10])
                        #     break
                        dcur = descs_np[ii, :, lind].reshape(-1)
                        for j in range(0, len(dcur)):
                            f_out.write(str(dcur[j]) + ' ')
                        f_out.write('\n')

            # if cnt % 100 == 0:
            #     curr_ap = get_ap(all_pos, all_neg)
            #     print(str(cnt/len(loader))+' ' + str(curr_ap))

        # curr_ap = get_ap(all_pos, all_neg)
        # print('final: ' + str(curr_ap)+' loss '+str(avg_loss/len(loader))+' time '+str(avg_time/len(loader)))
        # return curr_ap
        return 0


def net_inference(loader, encoder_net, save_descs=False, save_folder = ''):
    encoder_net.eval()
    with torch.no_grad():
        all_pos = []
        all_neg = []
        cnt = 0
        avg_loss = 0
        avg_time = 0
        for data in tqdm.tqdm(loader):
            x = Variable(data['images'].cuda(), requires_grad=False)
            if len(x) == 0:
                continue
            # print('bs='+str(len(x)))
            lines = Variable(data['lines'].cuda(), requires_grad=False)
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t0 = time.time()
            y = encoder_net(x)
            w_img = x[0].shape[2]
            h_img = x[0].shape[1]
            d = sample_descriptors(y, lines, w_img, h_img)

            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t1 = time.time()
            avg_time += t1-t0

            if save_descs:
                img_ids = data['image_ids']
                pair_ids = data['pair_ids']
                ln_lens = data['ln_lens']
                for ii in range(0, len(img_ids)):
                    img_id = img_ids[ii]
                    pair_id = pair_ids[ii]
                    f_out = open(save_folder + '/' + str(img_id) + '_' + str(pair_id) + '.txt', 'w')
                    descs_np = d.detach().cpu().numpy()
                    for lind in range(0, ln_lens[ii]):
                        dcur = descs_np[ii, :, lind].reshape(-1)
                        for j in range(0, len(dcur)):
                            f_out.write(str(dcur[j]) + ' ')
                        f_out.write('\n')
        return 0

