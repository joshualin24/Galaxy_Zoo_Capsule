import tensorflow as tf
from galaxy_CapsNet import *
import cv2
import numpy as np
import os
import sys
from galaxy_data import *
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1

batch_size = 32
num_class = 2
num_show = 5
is_multi_galaxy = True
is_shift_ag = True
irun = 0
is_show_multi_rec = False
is_show_sample = False
key = -1

# Build computaional graph
tf.reset_default_graph()
tf.set_random_seed(1234)

inputs = tf.placeholder(tf.float32, shape=(None,224,224,3))
target = tf.placeholder(tf.float32, [None, num_class])
dynamic_lr = tf.placeholder(tf.float32, shape=())

with slim.arg_scope(resnet_v1.resnet_arg_scope()):
   net, end_points = resnet_v1.resnet_v1_50(inputs, num_class, is_training=False)
   # net has shape (?,1,1,2), so use squeeze to reduce to (?,2)
   net = tf.squeeze(net)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net, labels=target))

prediction_prob = tf.nn.sigmoid(net)
tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

# Build Session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.33
sess = tf.Session(config=config)
writer = tf.summary.FileWriter("./sum",sess.graph)
var_to_save = [var for var in tf.global_variables() if ('Adam' not in var.name) and ('Momentum' not in var.name)]
saver = tf.train.Saver(var_list=var_to_save, max_to_keep=10)

sess.run(init)

RESTORE = True
if RESTORE:
    try:
        ckpt = tf.train.get_checkpoint_state('./cpt')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading model succeeded...')
    except:
        print('Loading model failed or model doest not exist')
        sys.exit()

GALAXY_TEST_LABEL_FILE = './data/test_samples/lens_parameters.txt'
GALAXY_TEST_FOLDER = './data/test_samples/'

GALAXY_TEST_RESULT_FILE = './data/test_samples/result.txt'

import pandas as pd

labels_df = pd.read_csv(GALAXY_TEST_LABEL_FILE, names=['elliptical','spiral'])
labels_elliptical = list(labels_df['elliptical'])
labels_spiral = list(labels_df['spiral'])

outF = open(GALAXY_TEST_RESULT_FILE, "w+")

# grid search from 0.1 to 0.9
grid_prob = [i/100.0 for i in range(10,100,10)]
# grid_prob = [0.5]

fp_counter = 0
fn_counter = 0

for prob_fp in grid_prob:
    for prob_fn in grid_prob:

        fp_counter = 0
        fn_counter = 0

        for i in range(100):
            filename = GALAXY_TEST_FOLDER + "img" + '_' + "%07d" % (i+1) + '.png'
            img = np.asarray(cv2.resize(cv2.cvtColor(cv2.imread(filename,0),cv2.COLOR_GRAY2RGB),(224,224))).reshape((224,224,3))
            logits, pred_length = sess.run([net, prediction_prob], feed_dict={inputs: [img]})
            #pred_length = pred_length[0]
            #print(pred_length)
            print('Elliptical - {:.2f}/{} | Spiral - {:.2f}/{}'.format(pred_length[0], labels_elliptical[i], pred_length[1], labels_spiral[i]))

            false_positive = False
            false_negative = False

            if labels_elliptical[i] == 0 and pred_length[0] > prob_fn:
                false_negative = True
            if labels_elliptical[i] == 1 and pred_length[0] < prob_fp:
                false_positive = True
            if labels_spiral[i] == 0 and pred_length[1] > prob_fn:
                false_negative = True
            if labels_spiral[i] == 1 and pred_length[1] < prob_fp:
                false_positive = True

            if false_positive:
                fp_counter += 1
            if false_negative:
                fn_counter +=1
            #outF.write('{}, {:.2f}, {}, {:.2f}, {}\n'.format(i+1,pred_length[0], labels_elliptical[i], pred_length[1], labels_spiral[i]))
        print('{:.2f},{:.2f},{},{}'.format(prob_fp,prob_fn,fp_counter,fn_counter))
        outF.write('{:.2f},{:.2f},{},{}'.format(prob_fp,prob_fn,fp_counter,fn_counter))
