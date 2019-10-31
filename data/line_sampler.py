import numpy as np
import torch
import time
def prepare_line_grid(lines_pair, margins, s, map_size, is_plain, pt_per_line):
    line_grid = prepare_grid_plain(lines_pair, margins, s, is_plain, pt_per_line)
    for i in range(0, 2):
        line_grid[:, :, :, i] -= 0.5 * map_size[i]
        line_grid[:, :, :, i] /= 0.5 * map_size[i]
    return line_grid

def sample_line(linedata, line_grid, i, li, s, margins, is_plain, pt_per_line):
    segs = []
    seg_num = 2
    if is_plain:
        seg_num = 1
    for si in range(0, seg_num):
        seg = linedata[4 * si: 4 * si + 4]
        if np.linalg.norm(seg) > 0:
            segs.append(seg)
    if len(segs) == 0:
        return
    seg_main = segs[0]
    x_s = np.float32(seg_main[0:2])
    x_e = np.float32(seg_main[2:4])
    dx = x_e - x_s
    pt_step = 1.0 / pt_per_line * dx
    # print 'line points start '+str(x_s) + ' end ' + str(x_e) + ' step ' + str(pt_step) + ' per line ' + str(self.pt_per_line)
    for pti in range(0, pt_per_line):
        pt_x = x_s + (0.5 + pti) * pt_step
        line_grid[i, li, pti, :] = s * pt_x - margins

def prepare_grid_vectorized_numpy(lines_pair, margins, s, pt_per_line):
    line_num = np.max([lines_pair[0].shape[1], lines_pair[1].shape[1]])
    line_grid = np.zeros((2, line_num, pt_per_line, 2))
    margins_vec = np.asarray(margins).reshape((2, 1))
    for i in range(0, 2):
        ld = lines_pair[i]
        cur_line_num = ld.shape[1]
        if cur_line_num == 0:
            continue
        x_s = ld[0:2,:]
        x_e = ld[2:4, :]
        for j in range(0, pt_per_line):
            c = (1.0+2*j)/(2*pt_per_line)
            m_rep = np.tile(margins_vec, (1, x_s.shape[1]))
            line_grid[i, 0:cur_line_num, j, :] = np.transpose(s*(x_s*(1-c)+ x_e*c) - m_rep, (1,0))

    return line_grid


def prepare_grid_numpy_vec(ld, s, pt_per_line):
    coordmats_lst = []
    cur_line_num = ld.shape[1]
    if cur_line_num == 0:
        return []
    x_s = ld[0:2,:]
    x_e = ld[2:4, :]
    pts_lst = []
    for j in range(0, pt_per_line):
        t0 = time.time()
        c = (1.0+2*j)/(2*pt_per_line)
        # m_rep = margins.view(2,1).expand(-1, x_s.shape[1])
        coordmat = s*(x_s*(1-c)+ x_e*c)# - m_rep
        # line_grid[i, 0:cur_line_num, j, :] = coordmat.permute(1,0)
        pts_lst.append(coordmat.transpose(1,0))
        t1= time.time()
        # print 'point compute takes '+str(t1-t0)
    return np.stack(pts_lst, axis=2).transpose(0,2,1)
    # allpts = torch.stack(pts_lst, dim = 2).permute(0,2,1)




def prepare_grid_vectorized(lines_pair, margins, s, pt_per_line):
    # line_num = torch.max([lines_pair[0].shape[1], lines_pair[1].shape[1]])
    # line_num = lines_pair.shape[2]
    t0 = time.time()
    # line_grid = torch.zeros((2, line_num, pt_per_line, 2)).float().cuda()
    t1 = time.time()
    # print 'line grid takes '+str(t1-t0)
    coordmats_lst = []
    for i in range(0, 2):
        ld = lines_pair[i]
        cur_line_num = ld.shape[1]
        if cur_line_num == 0:
            continue
        x_s = ld[0:2,:]
        x_e = ld[2:4, :]
        pts_lst = []
        for j in range(0, pt_per_line):
            t0 = time.time()
            c = (1.0+2*j)/(2*pt_per_line)
            # m_rep = margins.view(2,1).expand(-1, x_s.shape[1])
            coordmat = s*(x_s*(1-c)+ x_e*c)# - m_rep
            # line_grid[i, 0:cur_line_num, j, :] = coordmat.permute(1,0)
            pts_lst.append(coordmat.permute(1,0))
            t1= time.time()
            # print 'point compute takes '+str(t1-t0)
        allpts = torch.stack(pts_lst, dim = 2).permute(0,2,1)
        coordmats_lst.append(allpts)
    line_grid = torch.stack(coordmats_lst, dim=0)
    return line_grid

def prepare_grid_plain(lines_pair, margins, s, is_plain, pt_per_line):
    line_num = np.max([lines_pair[0].shape[1], lines_pair[1].shape[1]])
    line_grid = np.zeros((2, line_num, pt_per_line, 2))
    for i in range(0, 2):
        for li in range(0, lines_pair[i].shape[1]):
            linedata = lines_pair[i][:, li]
            sample_line(linedata, line_grid, i, li, s, margins, is_plain, pt_per_line)
    return line_grid
