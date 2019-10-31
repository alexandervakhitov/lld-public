# Learnable Line Descriptor Train and Inference demo

**Inference Installation instructions**

Please use Python 2.7. Setup a conda environment, install matplotlib, pytorch (tested with 0.4.1), OpenCV 3.0 and 
 our library [lbd_mod](https://github.com/alexandervakhitov/lbdmod.git), including the python interface, for line detection.

Download [archive](https://yadi.sk/d/lExOAYLNfF6i5w) and unzip it to the repo folder.

Run `python infer.py`

**Train Installation instructions**

Please use Python 2.7. Setup a conda environment, install matplotlib, pytorch (tested with 0.4.1), OpenCV 3.0 and 
 our library [lbd_mod](https://github.com/alexandervakhitov/lbdmod.git), including the python interface, for line detection.

Download [LLD Dataset](https://yadi.sk/d/D5QEuced7y5I1w) and unzip it to the '../batched' folder with respect to the cloned repo location.

Download [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and unzip it to the '../kitti' folder with respect to the cloned repo location.

Download [EuRoC MAV](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) zip files and unzip it to the '../euroc' folder with respect to the cloned repo location.

Download the EuRoC camera parameters file [EuRoC YAML](https://github.com/alexandervakhitov/lld-slam/blob/master/Examples/Stereo/EuRoC_Empty.yaml) and save it as '../EuRoC_Empty.yaml'.

Rectify EuRoC dataset using `python rectify_euroc.py ../euroc ../euroc_rect ../EuRoC_Empty.yaml` 

Create a folder `../kittieuroc`

Combine EuRoC and KITTI using `python prepare_kitti_euroc_combined.py ../kitti ../euroc_rect ../kittieuroc`

Run `python train.py`. It calls `train_multibatch` to train the network, `eval_multibatch` to evaluate and `test_with_descriptors_hetero` 
to save the descriptors to use with the [LLD-SLAM](https://github.com/alexandervakhitov/lld-slam.git). 


Please cite:  
A. Vakhitov, V. Lempitsky, Learnable Line Descriptor for Visual Navigation, *IEEE Access, 2019*
