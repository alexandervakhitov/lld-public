import os
import cv2
from torch.utils.data import Dataset
import numpy as np
import sys
import random
import torch
import line_sampler

def add_noise(im):
    return im.astype(float) + 30*np.random.randn(im.shape[0], im.shape[1])


def get_image_id(call_id, f1):
    mtd = 5
    main_id = 5 * call_id
    pair_id = -1
    img_id = -1
    if f1 < 4:
        pair_id = np.mod(f1, 2)
        img_id = np.floor(f1 / 2) + main_id + 5
        return int(img_id), pair_id
    else:
        fm = np.floor((f1 - 4) / 2)
        cnt = 0
        for j in range(1, mtd + 1):
            for k in range(-1, 2, 2):
                i = main_id + 5 + k * j
                if (i == main_id + 5 or i == main_id + 6):
                    continue
                if cnt == fm:
                    img_id = i
                    pair_id = np.mod(f1, 2)
                    return int(img_id), pair_id
                cnt += 1

    return -1, -1


def compose_batch(batch):
    # print('Composer!')
    batch = batch[0]
    n = len(batch['images'])

    ims = np.asarray(batch['images']).astype(float)
    # print('image range')
    # print(np.max(ims))
    # print(np.min(ims))
    ims = torch.from_numpy(ims).float()

    lines = batch['lines']
    ln_lens = [l.shape[0] for l in lines]



    if len(ln_lens)>0:
        ln_max = np.max(np.asarray(ln_lens))
        pt_num = lines[0].shape[1]
    else:
        ln_max = 0
        pt_num = 0
        n = 0
    lines_torch = torch.zeros(n, ln_max, pt_num, 2)
    for i in range(0, n):
        lines_torch[i, 0:lines[i].shape[0], :, :] = torch.from_numpy(lines[i].astype(float)).float()

    negs = batch['negatives']
    poss = batch['positives']
    if n > 0:
        n_pos = len(poss[0])
        max_neg = max([ni.shape[0] for ni in negs])
    else:
        max_neg = 0
        n_pos = 0
    neg_mask = -1*torch.ones(n, max_neg, n_pos).long()
    if len(negs)>0:
        for neg_i in negs:
            neg_mask[i][0:neg_i.shape[0]] = torch.from_numpy(neg_i).long()

    pos_t = -1*torch.ones(n, n_pos).long()
    for i in range(0, len(poss)):
        p = poss[i]
        for j in range(0, len(p)):
            if len(p[j]) == 1:
                pos_t[i, j] = p[j][0]

    fin_dict = batch
    fin_dict['images'] = ims
    fin_dict['lines'] = lines_torch
    fin_dict['negatives'] = neg_mask
    fin_dict['positives'] = pos_t
    fin_dict['ln_lens'] = ln_lens

    return fin_dict

def compose_infer_batch(batch):
    n = len(batch)
    ims = [b['images'] for b in batch]
    ims = np.asarray(ims).astype(float)
    ims = torch.from_numpy(ims).float()

    lines = [b['lines'] for b in batch]
    ln_lens = [l.shape[0] for l in lines]


    if len(ln_lens)>0:
        ln_max = np.max(np.asarray(ln_lens))
        pt_num = lines[0].shape[1]
    else:
        ln_max = 0
        pt_num = 0
        n = 0
    lines_torch = torch.zeros(n, ln_max, pt_num, 2)
    # print(len(lines))
    for i in range(0, n):
        lines_torch[i, 0:lines[i].shape[0], :, :] = torch.from_numpy(lines[i].astype(float)).float()

    fin_dict = {}
    fin_dict['images'] = ims
    fin_dict['lines'] = lines_torch
    fin_dict['ln_lens'] = ln_lens
    fin_dict['image_ids'] = [b['image_ids'] for b in batch]
    fin_dict['pair_ids'] = [b['pair_ids'] for b in batch]

    return fin_dict


def normalize_lines_for_gridsampler(im, li):
    w = im.shape[2]
    h = im.shape[1]
    l = li.astype(float)
    # print ('min - max x before ' + str(np.min(l[0, :]))+' ' + str(np.max(l[0, :])))
    l[0, :] = (l[0, :] - w / 2.0) / (w / 2.0)
    # print ('min - max x after ' + str(np.min(l[0, :])) + ' ' + str(np.max(l[0, :])))
    # print ('2 min - max x before ' + str(np.min(l[2, :])) + ' ' + str(np.max(l[2, :])))
    l[2, :] = (l[2, :] - w / 2.0) / (w / 2.0)

    # print ('2 min - max x after ' + str(np.min(l[2, :])) + ' ' + str(np.max(l[2, :])))
    l[1, :] = (l[1, :] - h / 2.0) / (h / 2.0)
    l[3, :] = (l[3, :] - h / 2.0) / (h / 2.0)
    return li


#Dataset description v1 (07.2018)
#lines: 5 x N matrix of ushort, with N number of lines in the image, and the rows are start_x, start_y, end_x, end_y, octave
#matching: N x M of uchar, where N is the number of detections in the frame, M is the number of tracks. Values are: 0 - not a match, 1 - possible match, 2 - detection from track

class DatasetBatchIndexer:

    def prepare_pyramid_listed(self, im):
        ims = []
        for i in range(0, self.pyramid_levels):
            ims.append(im)
            im = cv2.resize(im, (int(np.floor(1.0/self.resize_factor*im.shape[1]+0.5)), int(np.floor(1.0/self.resize_factor*im.shape[0]+0.5))))
        return ims


    def read_batch(self, data):
        frame_num = data.shape[0] / 8
        line_num = data.shape[1]
        # batch_data = np.zeros((2, 2, 2, line_num, frame_num), dtype=np.uint16)
        line_lst = []
        for li in range(0, line_num):
            line_projs = []
            for fi in range(0, frame_num):
                dets = []
                for di in range(0, 2):  # detection index
                    seg = data[8 * fi + 4 * di: 8 * fi + 4 * di + 4, li]
                    if np.linalg.norm(seg) > 0:
                        dets.append(seg)
                line_projs.append(dets)
            line_lst.append(line_projs)
        return line_lst

        # for pti in range(0, 2): #pt start - end
        #     for ci in range(0, 2):             #coordinate x-y
        #             batch_data[ci, pti, di, li, fi] = data[8*fi + 4*di + 2*pti + ci, li]
        # return batch_data

    def read_negatives(self, data):
        neg_data = []
        for fi in range(0, data.shape[0] / 8):
            frame_lines = []
            i = 0
            while i < data.shape[1] and np.linalg.norm(data[:, i]) > 0:
                line_seq = []
                for di in range(0, 2):
                    if np.linalg.norm(data[8 * fi + 4 * di: 8 * fi + 4 * di + 4, i]) == 0:
                        continue
                    pt_start = data[8 * fi + 4 * di: 8 * fi + 4 * di + 2, i]
                    pt_end = data[8 * fi + 4 * di + 2:8 * fi + 4 * di + 4, i]
                    seg_data = data[8 * fi + 4 * di: 8 * fi + 4 * di + 4, i]
                    # line_seq.append([pt_start, pt_end])
                    line_seq.append(seg_data)
                frame_lines.append(line_seq)
                i += 1
            neg_data.append(frame_lines)
        return neg_data

    def read_exceptions(self, data):
        fnum = data.shape[0] / 10
        batch_exc = []
        for fi in range(0, fnum):
            lnum = data.shape[1]
            frame_exc = []
            for li in range(0, lnum):
                ei = 0
                line_excs = []
                while ei < 10 and data[10 * fi + ei, li] > 0:
                    line_excs.append(data[10 * fi + ei, li])
                    ei += 1
                frame_exc.append(line_excs)
            batch_exc.append(frame_exc)
        return batch_exc

    def __init__(self, kitti_root_dir, root_dir, seq_start=-1, seq_end=-1, seq_inds = [], is_color = False, is_pyramid = False, is_careful = True,
                 is_report_dist = False, pt_per_line = 5, do_sample_points = True, is_add_noise=False):
        self.kitti_root_dir = kitti_root_dir
        self.root_dir = root_dir
        self.do_sample_points = do_sample_points
        self.is_color = is_color
        self.is_pyramid = is_pyramid
        self.is_careful = is_careful
        self.is_report_dist = is_report_dist
        self.pyramid_levels = 4
        self.resize_factor = 1.44
        self.pt_per_line = pt_per_line
        self.is_add_noise = is_add_noise


        self.positives = {}
        self.negatives = {}
        self.exceptions = {}

        self.exp_nums = {}
        self.exp_num = 0
        if len(seq_inds) == 0:
            seq_inds = np.arange(seq_start, seq_end+1)

        self.seq_inds = seq_inds
        for si in range(0, len(seq_inds)):
            seq_id = seq_inds[si]
            seq_dir = root_dir + '/' + str(seq_id) + '/'
            seq_data = {}
            neg_data = {}
            exc_data = {}
            cnt = 0
            flist = os.listdir(seq_dir)
            self.exp_nums[si] = len(flist)
            self.exp_num += len(flist)

    def compose_frame_data(self, seq_id, call_id, f1):
        pref = self.root_dir + '/' + str(seq_id) + '/' + str(call_id * 5) + '/' + str(f1)

        line_data = cv2.imread(pref + '_l.png', -1)
        if line_data is None:
            line_data = np.zeros((0,0))

        matches = cv2.imread(pref + '_m.png', -1)
        if matches is None:
            matches = np.zeros((0, 0))
        negs_all = []
        pos_all = []
        for i in range(0, matches.shape[1]):
            if self.is_careful:
                negs = np.nonzero(matches[:, i] == 255)[0]
            else:
                negs = np.nonzero(matches[:, i] != 2)[0]
            negs_all.append(negs)
            poss = np.nonzero(matches[:, i] == 2)[0]
            # print poss
            pos_all.append(poss)
        return line_data, negs_all, pos_all

    def compose_frame_data_w_mask(self, seq_id, call_id, f1):
        pref = self.root_dir + '/' + str(seq_id) + '/' + str(call_id * 5) + '/' + str(f1)

        line_data = cv2.imread(pref + '_l.png', -1)
        if line_data is None:
            line_data = np.zeros((0,0))

        matches = cv2.imread(pref + '_m.png', -1)
        if matches is None:
            matches = np.zeros((0, 0))
        negs_all = []
        pos_all = []
        negs = (matches == 255).astype(long) #N x n_p
        for i in range(0, matches.shape[1]):
            poss = np.nonzero(matches[:, i] == 2)[0]
            pos_all.append(poss)
        return line_data, negs, pos_all



    def get_label(self, i, tl):
        s = str(i)
        while len(s) < tl:
            s = '0' + s
        return s

    def get_image_gs(self, call_id, seq_id, f1):
        iid, pid = get_image_id(call_id, f1)
        # self.img_ids.append(iid)
        # self.pair_ids.append(pid)
        if iid < 0 or pid < 0:
            print 'error loading image'
        if self.is_color:
            pid = pid+2
        lbl = self.kitti_root_dir + '/' + self.get_label(seq_id, 2) + '/image_' + str(pid) + '/' + \
              self.get_label(iid,6) + '.png'
        if self.is_color:
            return cv2.imread(lbl)
        else:
            return cv2.imread(lbl, 0)
    def format_image(self, im1):
        im_size = [0,0]
        lu = [0,0]
        ims1 = []
        if self.is_pyramid:
            im_shape = im1.shape
            ims1 = self.prepare_pyramid_listed(im1)
            total_width = 0
            for im in ims1:
                total_width += im.shape[1]
            lus1 = []
            x = 0
            for i in range(0, len(ims1)):
                height = ims1[i].shape[0]
                width = ims1[i].shape[1]
                im_size = [height, width]
                lu = [0, x]
                x += width
        else:
            if self.is_color:
                im1 = np.transpose(im1, (2, 0, 1))
            else:
                im1 = im1.reshape((1, im1.shape[0], im1.shape[1]))
            im_size = im1.shape[1:3]
            lu = [0, 0]
            ims1 = im1
        return ims1, im_size, lu

    def format_images(self, im1, im2):
        ims = []
        im_sizes = []
        lus = []
        if self.is_pyramid:
            im_shape = im1.shape
            ims1 = self.prepare_pyramid_listed(im1)
            ims2 = self.prepare_pyramid_listed(im2)
            total_width = 0
            for im in ims1:
                total_width += im.shape[1]
            ims = np.zeros((2, im1.shape[0], total_width), dtype=np.uint8)
            lus1 = []
            lus2 = []
            x = 0
            for i in range(0, len(ims1)):
                height = ims1[i].shape[0]
                width = ims1[i].shape[1]
                ims[0, 0:height, x:x+width] = ims1[i]
                ims[1, 0:height, x:x+width] = ims2[i]
                im_sizes.append([height, width])
                lus1.append([0, x])
                lus2.append([0, x])
                x += width
            ims = np.asarray(ims)
            ims = ims.reshape((2, 1, ims.shape[1], ims.shape[2]))
            lus = [lus1, lus2]
        else:
            ims = [im1, im2]
            ims = np.asarray(ims)
            if self.is_color:
                ims = np.transpose(ims, (0, 3, 1, 2))
            else:
                ims = ims.reshape((2,1,ims.shape[1], ims.shape[2]))
            im_sizes.append(im1.shape[0:2])
            lus = [[0,0], [0,0]]
        return ims, im_sizes, lus

    def format_lines_for_image_pyr(self, l, lus, im_sizes):
        for i in range(0, l.shape[1]):
            oct_ind = l[4, i]
            if oct_ind > 0:
                lu = lus[oct_ind]
                k_h = float(im_sizes[oct_ind][0]) / im_sizes[0][0]
                k_w = float(im_sizes[oct_ind][1]) / im_sizes[0][1]
                l[0, i] = l[0, i] * k_w + lu[1]
                l[2, i] = l[2, i] * k_w + lu[1]
                l[1, i] = l[1, i] * k_h + lu[0]
                l[3, i] = l[3, i] * k_h + lu[0]
        return l

    def format_lines(self, l_lst, im_sizes, lus):
        if self.is_pyramid:
            for ii in range(0,2):
                l = l_lst[ii]
                l_lst[ii] = self.format_lines_for_image_pyr(l, lus[ii], im_sizes)
            return l_lst
        else:
            return l_lst

    def normalize_points_for_gridsampler(self, im, li):
        w = im.shape[2]
        h = im.shape[1]
        l = li.astype(float)
        l[:, 0, :] = (l[:, 0, :] - w / 2.0) / (w / 2.0)
        l[:, 1, :] = (l[:, 1, :] - h / 2.0) / (h / 2.0)
        # print ('min - max x before ' + str(np.min(l[0, :]))+' ' + str(np.max(l[0, :])))
        return l
    #        if do_sample_lines and not line_data.shape[0] == 0:

    # def form_neg_mask(self,all_negs):

#
    def get_multi_frame_batch(self, si, call_id, f_lst):
        seq_id = self.seq_inds[si]
        all_negs = []
        all_pos = []
        all_lines = []
        self.img_ids = []
        self.pair_ids = []
        ims = []
        ds = []
        for f in f_lst:
            iid, pid = get_image_id(call_id, f)
            self.img_ids.append(iid)
            self.pair_ids.append(pid)
            line_data, negs_1, pos_1 = self.compose_frame_data_w_mask(seq_id, call_id, f)
            if len(pos_1)==0:
                all_negs.append([])
                all_pos.append(None)
                all_lines.append([])
                ims.append([])
                ds.append(-1)
                continue
            im1 = self.get_image_gs(call_id, seq_id, f)
            if self.is_add_noise:
                im1 = add_noise(im1)
            im1, im_size1, lu1 = self.format_image(im1)
            if self.do_sample_points:
                line_data = line_sampler.prepare_grid_numpy_vec(line_data, 1.0, self.pt_per_line)
                line_data = self.format_lines(line_data, im_size1, lu1)
                # line_data = self.normalize_points_for_gridsampler(im1, line_data)
            else:
                # line_data = self.normalize_lines_for_gridsampler(im1, line_data)
                line_data = self.format_lines(line_data, im_size1, lu1)
            all_negs.append(negs_1)
            all_pos.append(pos_1)
            all_lines.append(line_data)
            ims.append(im1)
            ds.append(f)

        # all_inds = f_lst
        # print ('we sampled '+str(len(all_inds)))
        # print(all_inds)
        # rand_inds = np.random.choice(inds, n_lim-1, replace=False)
        # print(rand_inds)
        # all_inds = list(np.concatenate([np.zeros((1), dtype=int), rand_inds]))
        # def subfilter(lst, inds):
            # print inds
            # return [lst[i] for i in inds]
        fin_dict = {}
        fin_dict['images'] = ims
        # print(len(fin_dict['images']))
        fin_dict['lines'] = all_lines
        fin_dict['negatives'] = all_negs
        fin_dict['positives'] = all_pos
        # if self.is_report_dist:
        fin_dict['distances'] = ds
        fin_dict['image_ids'] = self.img_ids
        fin_dict['pair_ids'] = self.pair_ids

        #filter empty images
        p_fin = []
        neg_fin = []
        lines_fin = []
        ims_fin = []
        iids_fin = []
        pids_fin = []
        ds_fin = []
        for cnt in range(0, len(fin_dict['positives'])):
            p = fin_dict['positives'][cnt]
            if p is None:
                continue
            else:
                p_fin.append(p)
                neg_fin.append(fin_dict['negatives'][cnt])
                lines_fin.append(fin_dict['lines'][cnt])
                ims_fin.append(fin_dict['images'][cnt])
                iids_fin.append(fin_dict['image_ids'][cnt])
                pids_fin.append(fin_dict['pair_ids'][cnt])
                ds_fin.append(fin_dict['distances'][cnt])

        fin_dict['positives'] = p_fin
        fin_dict['negatives'] = neg_fin
        fin_dict['lines'] = lines_fin
        fin_dict['images'] = ims_fin
        fin_dict['image_ids'] = iids_fin
        fin_dict['pair_ids'] = pids_fin
        fin_dict['distances'] = ds_fin

        return fin_dict

    #we return
    #a) width-concatenated images in the array of size 2xCxWxH
    #b) line coords in a following format
    #   [L1, L2], where each Li is an array of size 5 x Ni, Ni is the number of lines, and each column is [sx, sy, ex, ey, ii],
    #   where (sx, sy) and (ex, ey) and line endpoints and ii is an image index to sample the line
    #c) pos_matches - a list of pairs such as (i,j) where i is an index of a line in L1, j is an index
    #   of a line in L2 and these lines should have close descriptors. If i or j empty, no match in this image
    #d) neg_matches - a list (N1, N2), each Ni is a list of negative matches, where at a a place k there is
    #   a list of negative matches for a positive pair number k in the list Li
    def get_dual_frame_batch(self, seq_id, call_id, f1, f2, dwn_factor=1.0):
        self.call_id = call_id
        self.f1 = f1
        self.f2 = f2
        t0 = cv2.getCPUTickCount()
        lines_1, negs_1, pos_1 = self.compose_frame_data(seq_id, call_id, f1)
        lines_2, negs_2, pos_2 = self.compose_frame_data(seq_id, call_id, f2)
        t1 = cv2.getCPUTickCount()
        pos_matches = []

        for p1i in range(0, len(pos_1)):
            p1 = pos_1[p1i]
            if len(pos_2) <= p1i:
                p2 = []
            else:
                p2 = pos_2[p1i]
            pos_matches.append([p1, p2])

        if not dwn_factor == 1.0:
            lines_1 = dwn_factor * lines_1
            lines_2 = dwn_factor * lines_2
        lines = [lines_1, lines_2]
        neg_matches = [negs_1, negs_2]

        t2 = cv2.getCPUTickCount()
        self.img_ids = []
        self.pair_ids = []
        self.seq_id = seq_id
        im1 = self.get_image_gs(call_id, seq_id, f1)
        im2 = self.get_image_gs(call_id, seq_id, f2)
        t3 = cv2.getCPUTickCount()
        dsize = (0,0)
        im1 = cv2.resize(im1, dsize, fx=dwn_factor, fy=dwn_factor, interpolation=cv2.INTER_LINEAR)
        im2 = cv2.resize(im2, dsize, fx=dwn_factor, fy=dwn_factor, interpolation=cv2.INTER_LINEAR)
        # ims, im_sizes, lus = self.format_images(im1, im2)
        ims1, im_size1, lu1 = self.format_image(im1)
        ims2, im_size2, lu2 = self.format_image(im2)
        ims = np.stack((ims1, ims2), axis=0)
        im_sizes = im_size1
        lus = [lu1, lu2]
        lines = self.format_lines(lines, im_sizes, lus)
        lines = [normalize_lines_for_gridsampler(ims[0], lines[0]),
                 normalize_lines_for_gridsampler(ims[1], lines[1])]
        if self.is_report_dist:
            return ims, lines, neg_matches, pos_matches, f2-f1, self.img_ids, self.pair_ids
        return ims, lines, neg_matches, pos_matches


class DatasetBatch(Dataset):

    def __init__(self, kitti_root_dir, root_dir, seq_inds =[], test_mode=False, is_color=False, is_pyramid=False,
                 is_careful=True, dwn_factor=1.0, is_rep_dist=False, batch_mode=False, full_pass=False, pt_per_line=5,
                 n_lim=6, is_noisy=False):
        self.indexer = DatasetBatchIndexer(kitti_root_dir, root_dir, seq_inds=seq_inds, is_color=is_color, is_pyramid=is_pyramid,
                                           is_careful=is_careful, is_report_dist=is_rep_dist, pt_per_line=pt_per_line, is_add_noise = is_noisy)
        self.dwn_factor = dwn_factor
        self.bs = 22-1
        self.exp_nums = {}
        self.exp_num = 0
        self.seq_ids = []
        self.test_mode = test_mode
        self.batch_mode = batch_mode
        self.full_pass = full_pass
        if batch_mode:
            self.n_lim = n_lim
            all_inds = np.arange(0, 22)
            n = int(len(all_inds)/self.n_lim) + 1
            self.part_inds = []
            for i in range(0, n):
                curr_ind = i * self.n_lim
                i_max = curr_ind+self.n_lim
                if i_max >= len(all_inds):
                    i_max = len(all_inds)
                self.part_inds.append(all_inds[curr_ind:i_max])

        for i in self.indexer.exp_nums:
            n = self.indexer.exp_nums[i]
            # if test_mode:
            self.exp_nums[i] = n * self.bs #* (self.bs - 1) / 2
            if self.batch_mode:
                self.exp_nums[i] = n
                if self.full_pass:
                    self.exp_nums[i] = n * len(self.part_inds)
                # else:
            #     self.exp_nums[i] = n
            self.exp_num += self.exp_nums[i]
            self.seq_ids.append(i)

    def __len__(self):
        if self.test_mode:
            return 100
        return self.exp_num

    def sample_multiframe_randomly(self):
        n_lim = self.n_lim
        inds = list(np.arange(1, self.bs+1))

        all_inds = [0]
        slotsize = len(inds) / (n_lim - 1) + 1
        # print(slotsize)
        for i in range(0, n_lim - 1):
            if len(all_inds) == n_lim:
                continue
            si = i * slotsize
            ei = (i + 1) * slotsize
            ei = min(ei, len(inds))
            cur_inds = inds[si:ei]
            # print(si)
            # print(ei)
            # print(cur_inds)
            new_ind = np.random.choice(cur_inds, 1)
            all_inds.append(new_ind[0])
        return all_inds

    def sample_multiframe_predefined(self, part_id):
        return self.part_inds[part_id]

    def __getitem__(self, idx):
        seq_cnt = 0
        seq_id = self.seq_ids[seq_cnt]
        agg_sum = 0
        while idx >= agg_sum:
            agg_sum += self.exp_nums[seq_id]
            seq_cnt += 1
            if seq_cnt < len(self.seq_ids):
                seq_id = self.seq_ids[seq_cnt]

        seq_cnt -= 1
        seq_id = self.seq_ids[seq_cnt]
        agg_sum -= self.exp_nums[seq_id]
        idx = idx - agg_sum
        if self.batch_mode:
            if self.full_pass:
                idx0 = int(idx /len(self.part_inds))
                part_id = idx - idx0 * len(self.part_inds)
                inds = self.sample_multiframe_predefined(part_id)
                mfb = self.indexer.get_multi_frame_batch(seq_id, idx0, inds)
            else:
                inds = self.sample_multiframe_randomly()
                mfb = self.indexer.get_multi_frame_batch(seq_id, idx, inds)
            if len(mfb['images']) == 0:
                if idx < len(self) - 1:
                    return self.__getitem__(idx + 1)
                else:
                    return self.__getitem__(0)
            else:
                return mfb
        else:

            # if self.test_mode:
            batch_data_len = self.bs # * (self.bs - 1) / 2
            call_id = int(np.floor(idx / batch_data_len))
            pair_id = idx - call_id * batch_data_len
            f1 = 0
            f2 = pair_id + 1
            return self.indexer.get_dual_frame_batch(seq_id, call_id, f1, f2, self.dwn_factor)
        # else:
        #     call_id = idx
        #     all_inds = list(np.arange(0, self.bs))
        #     f1, f2 = random.sample(all_inds, 2)
        #     return self.indexer.get_dual_frame_batch(seq_id, call_id, f1, f2)




def get_combined_training_v2(pt_per_line=5, n_lim=6, is_noisy=False):
    kitti_path = '../kittieuroc/'
    data_path = '../batched/'
    is_pyramid = False
    is_careful = True
    downsample_factor = 1.0
    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_dataset = DatasetBatch(kitti_path, data_path, seq_inds = [0,1,2,3,4,5,6,7,8,9,12,13,20,21],
                                 test_mode=False,
                                 is_color=False, is_pyramid=is_pyramid, is_careful=is_careful,
                                    dwn_factor=downsample_factor, batch_mode=True, pt_per_line=pt_per_line, n_lim=n_lim, is_noisy=is_noisy)  # 3,0
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               collate_fn=compose_batch,
                                               **kwargs)

    test_dataset = DatasetBatch(kitti_path, data_path, [10], test_mode=False, is_color=False, is_pyramid=is_pyramid,
                                is_careful=is_careful,
                                   dwn_factor=downsample_factor, batch_mode=True, pt_per_line=pt_per_line, n_lim=n_lim, full_pass=True, is_noisy=is_noisy)  # 3,0
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              collate_fn=compose_batch,
                                              **kwargs)
    test_dataset_2 = DatasetBatch(kitti_path, data_path, [11], False, False, is_pyramid, is_careful,
                                dwn_factor=downsample_factor, batch_mode=True, pt_per_line=pt_per_line, n_lim=n_lim,
                                full_pass=True)  # 3,0
    test_loader_2 = torch.utils.data.DataLoader(test_dataset_2,
                                              batch_size=1,
                                              shuffle=False,
                                              collate_fn=compose_batch,
                                              **kwargs)
    return train_loader, test_loader, test_loader_2


def get_combined_test(seq_id, pt_per_line=5, n_lim=6):
    kitti_path = '/media/hpc2_storage/avakhitov/kittieuroc/'
    data_path = '/media/hpc2_storage/avakhitov/kittieuroc/batched/'
    is_pyramid = False
    is_careful = True
    downsample_factor = 1.0
    kwargs = {'num_workers': 4, 'pin_memory': True}

    test_dataset = DatasetBatch(kitti_path, data_path, [seq_id], False, False, is_pyramid, is_careful,
                                   downsample_factor, batch_mode=True, pt_per_line=pt_per_line, n_lim=n_lim, full_pass=True)  # 3,0
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              collate_fn=compose_batch,
                                              **kwargs)
    return test_loader


class InferenceDataset(Dataset):

    def __init__(self, datadict, pt_per_line=5):
        self.datadict = datadict
        self.pt_per_line = pt_per_line

    def __len__(self):
        return len(self.datadict)

    def __getitem__(self, idx):
        img_fullpath, dets_fullpath, id, pair_id = self.datadict[idx]
        img = cv2.imread(img_fullpath, 0)
        img = np.reshape(img, (1, img.shape[0], img.shape[1]))
        lines = cv2.imread(dets_fullpath, -1)
        lines = line_sampler.prepare_grid_numpy_vec(lines, 1.0, self.pt_per_line)
        fin_dict = {}
        fin_dict['images'] = img
        # print(len(fin_dict['images']))
        fin_dict['lines'] = lines
        fin_dict['image_ids'] = id
        fin_dict['pair_ids'] = pair_id
        return fin_dict

