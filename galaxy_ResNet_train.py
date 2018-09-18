import tensorflow as tf
from galaxy_CapsNet import *
import cv2
import numpy as np
from galaxy_data import *
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1

batch_size = 20
num_class = 2
is_multi_galaxy = True
is_shift_ag = True
lr = 1e-3
steps = 100000
save_frequence = 1000
decay_frequence = 1000
min_lr = 1e-6

# Build computaional graph
tf.reset_default_graph()
tf.set_random_seed(1234)

inputs = tf.placeholder(tf.float32, shape=(None,224,224,3))
target = tf.placeholder(tf.float32, [None, num_class])
dynamic_lr = tf.placeholder(tf.float32, shape=())

with slim.arg_scope(resnet_v1.resnet_arg_scope()):
   net, end_points = resnet_v1.resnet_v1_50(inputs, num_class, is_training=True)
   # net has shape (?,1,1,2), so use squeeze to reduce to (?,2)
   net = tf.squeeze(net)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net, labels=target))
train = tf.train.AdamOptimizer(learning_rate=dynamic_lr).minimize(loss)

prediction_prob = tf.nn.sigmoid(net)
# let prob1 be [0.4,0.6], prob2 be [0.6,0.8]
# let target1 be [0,1], target2 be [1,0]
# then tf.abs(prediction_prob - target) for the 1st example is 0.8, and it is 1.2 for the 2nd example
# we can then set a ture/false threshold at 1
# then cast to int32 will change the output for the 1st example to 0, and for the 2nd example to be 1
# finally count nonzero elements, then divided by batch size
tmp = tf.reduce_sum(tf.abs(prediction_prob - target), axis=1)
error_rate = tf.cast(tf.count_nonzero(tf.cast(tmp, tf.int32),dtype=tf.int32),tf.float32) / tf.cast(tf.shape(target)[0], tf.float32)

tf.summary.scalar('error_rate_on_train_set', error_rate * 100.0)
tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

# Build Session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.33
sess = tf.Session(config=config)
writer = tf.summary.FileWriter("./sum",sess.graph)
var_to_save = [var for var in tf.global_variables() if ('Adam' not in var.name) and ('Momentum' not in var.name)]
# and ('beta1_power' not in var.name) and ('beta2_power' not in var.name)
saver = tf.train.Saver(var_list=var_to_save, max_to_keep=10)

sess.run(init)

RESTORE = True
if RESTORE:
    try:
        ckpt = tf.train.get_checkpoint_state('./cpt')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading model succeeded...')
    except:
        print("Loading model failed or model does't not exist")

# Train
irun = 0
for epoch in range(10):

    train_iter = multigalaxy_train_iter_alexnet(iters=steps,batch_size=batch_size,is_shift_ag = True)

    for X,Y,err in train_iter:

        if len(Y) != batch_size or err: 
            print("==========Done epoch %d!==========" % (epoch))    
            break
            
        LS, ACC, result, _ = sess.run([loss, error_rate, merged, train], feed_dict={inputs: X, target: Y, dynamic_lr: lr})
        ACC = 1- ACC
        writer.add_summary(result, irun)

        print('{}, LS: {:.4f}, ACC:{:.3f}'.format(irun, LS, ACC))

        if (irun+1) % save_frequence == 0:
            saver.save(sess, './cpt/resnet_v1_50-{}.ckpt'.format(irun+1))

        if (irun+1) % decay_frequence == 0 and lr > min_lr:
            lr *= 0.5

        irun += 1
