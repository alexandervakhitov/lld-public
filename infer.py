import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pylbd
import matplotlib.pyplot as plt
import torch.nn as nn

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

def sample_descriptors(x, lines, w_img, h_img):
    w_map = x.shape[3]
    h_map = x.shape[2]
    # print(x.shape)
    m_x = (w_img-w_map)/2.0
    m_y = (h_img-h_map)/2.0
    # print(w_img)
    # print(m_x)
    lines_x_flat = lines[:, :, :, 0].contiguous().view(-1)
    good_lines = (lines_x_flat > 0).nonzero()
    # print(str(len(good_lines))+ ' ' + str(lines_x_flat.shape[0]))
    lines_x_flat[good_lines] = lines_x_flat[good_lines] + m_x*torch.ones_like(lines_x_flat[good_lines])
    lines_x_flat[good_lines] = 2.0/w_map*lines_x_flat[good_lines] - torch.ones_like(lines_x_flat[good_lines])
    lines[:, :, :, 0] = lines_x_flat.view(lines[:, :, :, 0].shape)

    lines_y_flat = lines[:, :, :, 1].contiguous().view(-1)
    good_lines = (lines_y_flat > 0).nonzero()
    lines_y_flat[good_lines] = lines_y_flat[good_lines] + m_y*torch.ones_like(lines_y_flat[good_lines])
    lines_y_flat[good_lines] = 2.0 / h_map * lines_y_flat[good_lines] - torch.ones_like(lines_y_flat[good_lines])
    lines[:, :, :, 1] = lines_y_flat.view(lines[:, :, :, 1].shape)

    lds = F.grid_sample(x, lines, mode='bilinear', padding_mode='border')
    avg_lds = torch.sum(lds, 3)
    avg_lds = F.normalize(avg_lds, p=2, dim=1)
    return avg_lds

def prepare_grid_numpy_vec(ld, s, pt_per_line):
    cur_line_num = ld.shape[1]
    if cur_line_num == 0:
        return []
    x_s = ld[0:2,:]
    x_e = ld[2:4, :]
    pts_lst = []
    for j in range(0, pt_per_line):
        c = (1.0+2*j)/(2*pt_per_line)
        coordmat = s*(x_s*(1-c)+ x_e*c)# - m_rep
        pts_lst.append(coordmat.transpose(1,0))
    return np.stack(pts_lst, axis=2).transpose(0,2,1)

def prepare_input(img, lines, is_cuda):
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
    img = np.asarray(img).astype(float)
    img = torch.from_numpy(img).float()
    lines = lines[:, 7:11].transpose()
    lines = prepare_grid_numpy_vec(lines, 1.0, pt_per_line=5)
    lines = lines.reshape(1, lines.shape[0], lines.shape[1], lines.shape[2])
    lines = torch.from_numpy(lines).float()
    if is_cuda:
        img = img.cuda()
        lines = lines.cuda()
    return img, lines

def match_using_lbd(img1, img2, n_oct, factor):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    lbd1, lines1 = pylbd.detect_and_describe(gray1, n_oct, factor)
    lbd2, lines2 = pylbd.detect_and_describe(gray2, n_oct, factor)
    matches = pylbd.match_lbd_descriptors(lbd1, lbd2)
    debug_lbd = pylbd.visualize_line_matching(img1, lines1, img2, lines2, matches, True)
    return debug_lbd

def test_line_matching(weights_path, is_cuda):
    n_oct = 1
    factor = 1.44
    img1 = cv2.imread('kitti_8_left.png')
    img2 = cv2.imread('kitti_8_right.png')
    w_img = img1.shape[1]
    h_img = img1.shape[0]
    debug_lbd = match_using_lbd(img1, img2, n_oct, factor)
    cv2.imwrite('test_lbd.png', debug_lbd)
    plt.figure('LBD')
    plt.imshow(debug_lbd)

    encoder_net = FeatureEncoder(is_cuda=True, is_color=False, is_pyramid = False, depth=3, g=0)
    if is_cuda:
        encoder_net =  encoder_net.cuda()
    checkpoint = torch.load(weights_path)
    encoder_net.load_state_dict(checkpoint['state_dict'])

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    lines1 = pylbd.detect_edlines(gray1, n_oct, factor)
    img1_torch, lines1_torch = prepare_input(gray1, lines1, is_cuda)
    y = encoder_net(img1_torch)
    d1 = sample_descriptors(y, lines1_torch, w_img, h_img).detach().cpu().numpy()

    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    lines2 = pylbd.detect_edlines(gray2, n_oct, factor)
    img2_torch, lines2_torch = prepare_input(gray2, lines2, is_cuda)
    y2 = encoder_net(img2_torch)
    d2 = sample_descriptors(y2, lines2_torch, w_img, h_img).detach().cpu().numpy()

    nd1 = d1.shape[2]
    nd2 = d2.shape[2]
    match_lst = []
    d1 = d1.reshape(64, -1).transpose()
    d2 = d2.reshape(64, -1).transpose()
    for i in range(0, nd1):
        ld1 = d1[i]
        min_dist = 1e10
        best_match = 0
        for j in range(0, nd2):
            ld2 = d2[j]
            dist = 2 - ld1.dot(ld2)
            if dist < min_dist:
                min_dist = dist
                best_match = j

        match_result = [i, best_match, 0, 1]
        match_lst.append(match_result)
    matches_lld = np.asarray(match_lst).astype(int)

    debug_lld_img = pylbd.visualize_line_matching(img1, lines1, img2, lines2, matches_lld, True)
    cv2.imwrite('test_lld.png', debug_lld_img)
    plt.figure('LLD')
    plt.imshow(debug_lld_img)
    plt.show()


test_line_matching(weights_path = '/storage/projects/lld/1.pyh.tar', is_cuda = False)