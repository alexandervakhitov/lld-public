import os
import shutil

def combine_datasets(kitti_path, euroc_rect_path, target_path):

    ds_def = [(0,0),
              (0,1),
              (0,2),
              (0,3),
              (0,4),
              (1,0),
              (1,1),
              (1,3),
              (1,5),
              (1,7),
              (0,7),
              (1,9),
              (0,6),
              (0,5),
              (0,8),
              (0,9),
              (0,10),
              (1,2),
              (1,4),
              (1,6),
              (1,8),
              (1,10)]

    for i, ds in enumerate(ds_def):
        sc = ds[1]
        lbl = str(sc)
        if len(lbl) < 2:
            lbl = '0' + lbl
        tgt_lbl = str(i)
        if len(tgt_lbl) < 2:
            tgt_lbl = '0' + tgt_lbl
        if ds[0] == 0: #kitti
            shutil.copytree(kitti_path + '/' + lbl + '/',
                            target_path+ '/' + tgt_lbl)
        else:
            shutil.copytree(euroc_rect_path + '/' + lbl + '/',
                            target_path + '/' + tgt_lbl)

if __name__ == '__main__':
    kitti_path = sys.argv[1]
    euroc_rect_path = sys.argv[2]
    out_path = sys.argv[3]
    combine_datasets(kitti_path,
                 euroc_rect_path,
                 out_path)


