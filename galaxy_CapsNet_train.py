import tensorflow as tf
from galaxy_CapsNet import *
import cv2
import numpy as np
from galaxy_data import *

batch_size = 32
num_show = 5
is_multi_galaxy = True
is_shift_ag = True
irun = 0
lr = 1e-3
steps = 100000
save_frequence = 2500
decay_frequence = 5000
is_show_multi_rec = False
is_show_sample = False
key = -1
min_lr = 5e-6

if is_multi_galaxy:
    train_iter = multigalaxy_train_iter(iters=steps,batch_size=batch_size,is_shift_ag = True)
    test_iter = multigalaxy_test_iter(iters=steps, batch_size=batch_size,is_shift_ag = True)
else:
    train_iter = galaxy_train_iter(iters=steps,batch_size=batch_size,is_shift_ag = True)
    test_iter = galaxy_test_iter(iters=steps, batch_size=batch_size,is_shift_ag = True)
multi_iter = multigalaxy_test_iter(iters=steps,batch_size=num_show,is_shift_ag = True)

tf.reset_default_graph()
tf.set_random_seed(1234)
net = CapsNet(is_multi_galaxy=is_multi_galaxy)
tf.summary.scalar('error_rate_on_test_set', (1.0 - net.accuracy) * 100.0)
tf.summary.scalar('loss_reconstruction_on_test_set', net.loss_rec)
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.33
sess = tf.Session(config=config)
writer = tf.summary.FileWriter("./sum",sess.graph)
var_to_save = [var for var in tf.global_variables() if ('Adam' not in var.name) and ('Momentum' not in var.name)]
saver = tf.train.Saver(var_list=var_to_save, max_to_keep=10)

sess.run(init)

RESTORE = False
if RESTORE:
    try:
        ckpt = tf.train.get_checkpoint_state('./cpt')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading model succeeded...')
    except:
        print('Loading model failed or model doest not exist')

for X,Y in train_iter:

    X_TEST, Y_TEST = test_iter.__next__()

    # LS, LS_REC, ACC, _ = sess.run([net.loss, net.loss_rec, net.accuracy, net.train], feed_dict={net.x: X, net.y: Y, net.lr: lr, net.is_training:True})
    LS, ACC, _ = sess.run([net.loss, net.accuracy, net.train], feed_dict={net.x: X, net.y: Y, net.lr: lr, net.is_training:True})
    ACC_TEST, result = sess.run([net.accuracy,merged], feed_dict={net.x: X_TEST, net.y: Y_TEST, net.is_training:False})

    writer.add_summary(result, irun)

    # print('{},{},{},{},{}'.format(irun, LS, LS_REC, ACC, ACC_TEST))
    print('{}, LS: {:.4f}, Train ACC:{:.3f}, Test ACC: {:.3f}'.format(irun, LS, ACC, ACC_TEST))

    """
    if is_show_sample:
        H_SAM = np.random.randn(num_show*2, 2, 32)
        H_SAM = H_SAM / (0.001 + np.sum(H_SAM ** 2.0, axis=-1, keepdims=True) ** 0.5)
        Y_SAM = np.eye(2)[np.array(list(range(2)) * num_show)].astype(float)
        X_SAM = sess.run(net.x_sample, feed_dict={net.h_sample: H_SAM, net.y_sample: Y_SAM, net.is_training:False})
        images_sample = X_SAM.reshape([num_show, 2, 212, 212, 1])
        images_sample = np.concatenate(images_sample, axis=1)
        images_sample = cv2.resize(np.concatenate(images_sample, axis=1), dsize=(0, 0), fx=3, fy=3)
        cv2.imshow('SampleFromH', images_sample)

    if is_show_multi_rec:
        X_MULTI,Y_MULTI = multi_iter.__next__()
        X_REC1,X_REC2 = sess.run(net.x_recs, feed_dict={net.x: X_MULTI, net.y: Y_MULTI, net.is_training:False})
        # turn the composed image to be 3 channel gray image
        images_org = np.stack([X_MULTI[:num_show,:,:,0]]*3,axis=-1)
        black = np.zeros([num_show, 212, 212, 1])
        images_recs = np.concatenate([black, X_REC1, X_REC2], axis=-1)
        images_rec1 = np.concatenate([black, black, X_REC2], axis=-1)
        images_rec2 = np.concatenate([black, X_REC1, black], axis=-1)
        image_show = np.concatenate([images_org, images_recs, images_rec1, images_rec2], axis=2)
        image_show = cv2.resize(np.concatenate(image_show, axis=0), dsize=(0, 0), fx=3, fy=3)
        cv2.imshow('MultigalaxyReconstruction', image_show)

    if is_show_multi_rec or is_multi_galaxy:
        key = cv2.waitKey(1)

    if key == ord('s'):
        cv2.imwrite('MultigalaxyReconstruction%d.png' % irunloss_reconstruction_on_test_set, image_show * 255.0)
        cv2.imwrite('SampleFromH%d.png' % irun, images_sample * 255.0)
    """

    if (irun+1) % save_frequence == 0:
        saver.save(sess, './cpt/model2-{}.ckpt'.format(irun+1))

    if (irun+1) % decay_frequence == 0 and lr > min_lr:
        lr *= 0.5

    irun += 1
