import tensorflow as tf
from tensorflow.python.training import moving_averages
import numpy as np

class CapsNet(object):
    def __init__(self,routing_iterations = 3,batch_size=32,is_multi_galaxy=False,beta1=0.9):

        self.iterations = routing_iterations
        self.batch_size = batch_size
        self.is_multi_galaxy = float(is_multi_galaxy)
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, [None, 212, 212, 3])
        self.h_sample = tf.placeholder(tf.float32, [None, 2, 32])
        self.y_sample = tf.placeholder(tf.float32, [None, 2])
        self.y = tf.placeholder(tf.float32, [None, 2, 3])
        self.lr = tf.placeholder(tf.float32)
        self._extra_train_ops = []

        x_composed, x_a, x_b = tf.split(self.x,num_or_size_splits=3,axis=3)
        y_composed, y_a, y_b = tf.split(self.y,num_or_size_splits=3,axis=2)

        valid_mask = self.is_multi_galaxy * (tf.reduce_sum(y_composed, axis=[1,2])) \
                      + (1.0 - self.is_multi_galaxy) * tf.ones_like(y_composed[:,0,0])

        v_digit,c = self.get_CapsNet(x_composed)
        self.length_v = tf.reduce_sum(v_digit ** 2.0, axis=-1) ** 0.5  # self.length_v with shape [batch_size,10]

        x_rec_a = self.get_deconv_decoder(v_digit * y_a)
        x_rec_b = self.get_deconv_decoder(v_digit * y_b,reuse=True)
        loss_rec_a = tf.reduce_sum((x_rec_a - x_a) ** 2.0, axis=[1, 2, 3])
        loss_rec_b = tf.reduce_sum((x_rec_b - x_b) ** 2.0, axis=[1, 2, 3])
        self.loss_rec = (loss_rec_a + loss_rec_b) / 2.0
        self.x_recs = [x_rec_a,x_rec_b]
        self.x_sample = self.get_deconv_decoder(self.h_sample * self.y_sample[:, :, None], reuse=True)
        self.loss_cls = tf.reduce_sum(y_composed[:,:,0] * tf.maximum(0.0, 0.9 - self.length_v) ** 2.0
                                      + 0.5 * (1.0 - y_composed[:,:,0]) * tf.maximum(0.0, self.length_v - 0.1) ** 2.0,axis=-1)
        self.loss_cls = tf.reduce_sum(self.loss_cls*valid_mask)/tf.reduce_sum(valid_mask)
        self.loss_rec = tf.reduce_sum(self.loss_rec*valid_mask)/tf.reduce_sum(valid_mask)
        self.loss = self.loss_cls # + 0.0005*self.loss_rec
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=beta1)
        grads_vars= self.optimizer.compute_gradients(self.loss)
        apply_op = self.optimizer.apply_gradients(grads_vars)
        train_ops = [apply_op] + self._extra_train_ops
        # Group all updates to into a single train op.
        self.train = tf.group(*train_ops)

        if is_multi_galaxy:
            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.length_v,tf.argmax(tf.squeeze(y_a), 1),k=1),tf.float32))+\
                            tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.length_v,tf.argmax(tf.squeeze(y_b), 1),k=1),tf.float32))
            self.accuracy /= 2.0
            #this may be different from the paper
        else:
            correct_prediction = tf.equal(tf.argmax(y_composed[:,:,0], 1), tf.argmax(self.length_v, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def get_CapsNet(self,x,reuse = False):

        with tf.variable_scope('init'):
            # x has shape [-1,212,212,3],
            # deconvlution output should be [-1,212,212,1] instead of [-1,212,212,3]
            self.deconvlution_shape = [x.get_shape().as_list()[1:-1]+[1]]
            x = self._batch_norm('init_bn', x)
            x = self._conv('init_conv', x, 5, 1, 128, self._stride_arr(3), padding='VALID')
            x = self._relu(x)
            tf.logging.info('image after init {}'.format(x.get_shape()))
            self.deconvlution_shape = [x.get_shape().as_list()[1:]] + self.deconvlution_shape

        with tf.variable_scope('init2'):
            x = self._batch_norm('init_bn', x)
            x = self._conv('init_conv', x, 5, 128, 256, self._stride_arr(3), padding='VALID')
            x = self._relu(x)
            tf.logging.info('image after init {}'.format(x.get_shape()))
            self.deconvlution_shape = [x.get_shape().as_list()[1:]] + self.deconvlution_shape

        with tf.variable_scope('primal_capsules'):
            x = self._batch_norm('primal_capsules_bn', x)
            x = self._conv('primal_capsules_conv', x, 5, 256, 256, self._stride_arr(3), padding='VALID')

            capsules_dims = 16
            num_capsules = np.prod(x.get_shape().as_list()[1:]) // capsules_dims
            self.deconvlution_shape = [x.get_shape().as_list()[1:]] + self.deconvlution_shape

            x = tf.reshape(x, [-1, num_capsules, capsules_dims])
            x = self._squash(x, axis=-1)
            tf.logging.info('image after primal capsules {}'.format(x.get_shape()))

        with tf.variable_scope('digital_capsules_1'):
            """
                params_shape = [input_num_capsule,input_dim_capsule
                                output_num_capsule,output_dim_capsule]
            """
            params_shape = [num_capsules, capsules_dims, 64, 8]
            x,_ = self._capsule_layer(x, params_shape=params_shape,
                                         num_routing=self.iterations, name='digital_capsule')

        with tf.variable_scope('digital_capsules_2'):
            """
                params_shape = [input_num_capsule,input_dim_capsule
                                output_num_capsule,output_dim_capsule]
            """
            params_shape = [64, 8, 2, 32]
            v, c = self._capsule_layer(x, params_shape=params_shape,
                                         num_routing=self.iterations, name='digital_capsule')
        return v, c

    def get_deconv_decoder(self,h,reuse=False):
        batch_size = tf.shape(h)[0]
        if batch_size is None:
            batch_size = self.batch_size
        h = tf.reshape(h,[-1,2*32])
        with tf.variable_scope('decoder',reuse=reuse):
            h = self._fully_connected(h, 64*8, name='fc1')
            h = self._fully_connected(h, np.prod(self.deconvlution_shape[0]), name='fc2')
            h = tf.reshape(h, [batch_size]+self.deconvlution_shape[0])

            kernel1 = tf.get_variable('DW1', [5,5,256,256],tf.float32, initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / (256+256))))
            h = tf.nn.conv2d_transpose(h, kernel1, output_shape=tf.stack([batch_size]+self.deconvlution_shape[1]), strides=[1,3,3,1],padding='VALID')

            kernel2 = tf.get_variable('DW2', [5,5,128,256],tf.float32, initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / (256+128))))
            h = tf.nn.conv2d_transpose(h, kernel2, output_shape=tf.stack([batch_size]+self.deconvlution_shape[2]), strides=[1,3,3,1],padding='VALID')

            kernel3 = tf.get_variable('DW3', [5,5,1,128],tf.float32, initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / (128))))
            h = tf.nn.conv2d_transpose(h, kernel3, output_shape=tf.stack([batch_size]+self.deconvlution_shape[3]), strides=[1,3,3,1],padding='VALID')
        return h

    def _capsule_layer(self, x, params_shape, num_routing, name=''):

        assert len(params_shape) == 4, "Given wrong parameter shape."
        input_num_capsule, input_dim_capsule, output_num_capsule, output_dim_capsule = params_shape

        # x = self._batch_norm(name + '/bn', x)

        # W.shape =  [None, input_num_capsule, input_dim_capsule, output_num_capsule, output_dim_capsule]
        W = tf.get_variable(
            name + '/capsule_layer_transformation_matrix', [
                1, input_num_capsule, input_dim_capsule, output_num_capsule, output_dim_capsule], tf.float32,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))

        # b.shape = [None, self.intput_num_capsule, 1, self.output_num_capsule, 1].
        b = tf.zeros([1, input_num_capsule, 1, output_num_capsule, 1])
        c = tf.nn.softmax(b, dim=3)

        # u.shape=[None, input_num_capsule, input_dim_capsule, 1, 1]
        u = tf.expand_dims(tf.expand_dims(x, -1), -1)
        u_ = tf.reduce_sum(u * W, axis=[2], keep_dims=True)
        u_stopped = tf.stop_gradient(u_)

        s = tf.reduce_sum(u_stopped * c, axis=[1], keep_dims=True)
        v = self._squash(s, axis=-1)
        tf.logging.info('Expanding inputs to be {}'.format(u.get_shape()))
        tf.logging.info(
            'Transforming and sum input capsule dimension (routing inputs){}'.format(u_.get_shape()))
        tf.logging.info('Outputs of each routing iteration {}'.format(v.get_shape()))

        assert num_routing > 1, 'The num_routing should be > 1.'

        for i in range(num_routing - 2):
            b += tf.reduce_sum(u_stopped * v, axis=-1, keep_dims=True)
            c = tf.nn.softmax(b, dim=3)
            s = tf.reduce_sum(u_stopped * c, axis=[1], keep_dims=True)
            v = self._squash(s, axis=-1)

        b += tf.reduce_sum(u_ * v, axis=-1, keep_dims=True)
        c = tf.nn.softmax(b, dim=3)
        s = tf.reduce_sum(u_ * c, axis=[1], keep_dims=True)
        v = self._squash(s, axis=-1)

        v_digit = tf.squeeze(v, axis=[1, 2])
        tf.logging.info('Image after this capsule layer {}'.format(v_digit.get_shape()))
        return v_digit, c

    def _squash(self,s, axis=-1):
        length_s = tf.reduce_sum(s ** 2.0, axis=axis, keep_dims=True) ** 0.5
        v = s * length_s / (1.0 + length_s ** 2.0)
        return v

    """Anxilliary methods"""

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
            moving_mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(
                    moving_mean, mean, 0.99))
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(
                    moving_variance, variance, 0.99))

            def train():
                # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
                return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)

            def test():
                return tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, 0.001)
            y = tf.cond(tf.equal(self.is_training, tf.constant(True)), train, test)
            y.set_shape(x.get_shape())
            return y

    # override _conv to use He initialization with truncated normal to prevent dead neural
    def _conv(self, name, x, filter_size, in_filters, out_filters, strides, padding='VALID'):
        """Convolution."""
        with tf.variable_scope(name):
            # n = filter_size * filter_size * out_filters
            n = in_filters + out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.truncated_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding=padding)

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _relu(self, x, leakiness=0.0, elu=False):
        """Relu, with optional leaky support."""
        if leakiness > 0.0:
            return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
        elif elu:
            return tf.nn.elu(x)
        else:
            return tf.nn.relu(x)

    def _fully_connected(self, x, out_dim, name='', dropout_prob=None):
        """FullyConnected layer for final output."""
        x = tf.contrib.layers.flatten(x)

        if dropout_prob is not None:
            x = tf.nn.dropout(x, keep_prob=dropout_prob)

        w = tf.get_variable(
            name + 'DW', [x.get_shape()[1], out_dim],
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
        b = tf.get_variable(name + 'biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
