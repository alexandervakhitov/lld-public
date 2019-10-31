import cv2
import os
import sys

def prepare_rectifier(conf_path):
    fs = cv2.FileStorage(conf_path, cv2.FILE_STORAGE_READ)
    Kl = fs.getNode("LEFT.K").mat()
    Dl = fs.getNode("LEFT.D").mat()
    Rl = fs.getNode("LEFT.R").mat()
    Pl = fs.getNode("LEFT.P").mat()
    wl = 752
    hl = 480

    Kr = fs.getNode("RIGHT.K").mat()
    Dr = fs.getNode("RIGHT.D").mat()
    Rr = fs.getNode("RIGHT.R").mat()
    Pr = fs.getNode("RIGHT.P").mat()
    wr = 752
    hr = 480


#cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,*M1l_p,*M2l_p);
    M1l, M2l = cv2.initUndistortRectifyMap(Kl, Dl, Rl, Pl[0:3, 0:3], (wl, hl), cv2.CV_32F)
    M1r, M2r = cv2.initUndistortRectifyMap(Kr, Dr, Rr, Pr[0:3, 0:3], (wr, hr), cv2.CV_32F)

    return M1l, M2l, M1r, M2r

def rectify_sequence(seq_path, out_path, M1, M2):
    cnt = 0
    for f in sorted(os.listdir(seq_path)):
        if '.png' in f:
            img = cv2.imread(seq_path + '/' + f)
            img_out = cv2.remap(img, M1, M2, cv2.INTER_LINEAR)
            lbl = str(cnt)
            while len(lbl) < 6:
                lbl = '0' + lbl
            cv2.imwrite(out_path + '/' + lbl + '.png', img_out)
            cnt += 1

def rectify_dataset(M1l, M2l, M1r, M2r, path_to_dataset, path_to_rect):
    seqs = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult", "V2_01_easy", "V2_02_medium", "V2_03_difficult"]
    for si, s in enumerate(seqs):
        if si < 5:
            continue
        slbl = str(si)
        if len(slbl) < 2:
            slbl = '0' + slbl
        path_to_left = path_to_dataset + '/' + s + '/mav0/cam0/data/'
        path_to_left_rect = path_to_rect + '/' + slbl + '/image_0/'
        if not os.path.exists(path_to_left_rect):
            os.makedirs(path_to_left_rect)
        rectify_sequence(path_to_left, path_to_left_rect, M1l, M2l)

        path_to_right = path_to_dataset + '/' + s + '/mav0/cam1/data/'
        path_to_right_rect = path_to_rect + '/' + slbl + '/image_1/'
        if not os.path.exists(path_to_right_rect):
            os.makedirs(path_to_right_rect)
        rectify_sequence(path_to_right, path_to_right_rect, M1r, M2r)


if __name__ == '__main__':
    euroc_path = sys.argv[1]
    out_path = sys.argv[2]
    conf_path = sys.argv[3]
    M1l, M2l, M1r, M2r = prepare_rectifier(conf_path)
    rectify_dataset(M1l, M2l, M1r, M2r, euroc_path, out_path)
