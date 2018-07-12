import tensorflow as tf
from galaxy_CapsNet import *
import cv2
import os
import sys
import numpy as np
from galaxy_data import *

batch_size = 10
is_multi_galaxy = True
is_shift_ag = True
irun = 0
lr = 1e-3
steps = 100

if is_multi_galaxy:
    train_iter = multigalaxy_train_iter(iters=steps,batch_size=batch_size,is_shift_ag = True)
    test_iter = multigalaxy_test_iter(iters=steps, batch_size=batch_size,is_shift_ag = True)
else:
    train_iter = galaxy_train_iter(iters=steps,batch_size=batch_size,is_shift_ag = True)
    test_iter = galaxy_test_iter(iters=steps, batch_size=batch_size,is_shift_ag = True)

folder = './data/test_samples/'
if not os.path.exists(folder):
    os.mkdir(folder)

outF = open(folder+ "lens_parameters.txt", "w+")
i = 0
for X,Y in train_iter:
    for imgs in X:
        imgs_T = np.transpose(imgs, (2, 0, 1))
        img_name = "img" + '_' + "%07d" % (i+1) + '.png'
        cv2.imwrite(folder+img_name, imgs_T[0])
        i = i + 1
    for labels in Y:
        outF.write(str(int(labels[0][0]))+','+str(int(labels[1][0]))+'\n')
