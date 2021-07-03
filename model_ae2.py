import tensorflow as tf
import numpy as np

def weight_variable(shape, scope=None, name=None):
    with tf.variable_scope(name_or_scope=scope, default_name=name):
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.001)
        return tf.Variable(initial)


def bias_variable(shape, scope=None, name=None):
    with tf.variable_scope(name_or_scope=scope, default_name=name):
        initial = tf.constant(0.0, shape=shape, dtype=tf.float32, name=name)
        return tf.Variable(initial)


def conv2d(x, w, strides, name=None):
    return tf.nn.conv2d(x, w, strides=[1, strides[0], strides[1], 1], padding="SAME", name=name)


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)


def deconv(x, w, output_shape, strides, name=None):
    dyn_input_shape = tf.shape(x)
    batch_size = dyn_input_shape[0]
    output_shape = tf.stack([batch_size, output_shape[1], output_shape[2], output_shape[3]])
    output = tf.nn.conv2d_transpose(x, w, output_shape, strides, padding="SAME", name=name)
    return output


def encoder(x1, x2):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        chn_base = 16
        # shape is [batch, 31, 72, 1]
        # Encoder 1
        w_conv = weight_variable([3, 3, 1, chn_base], 'w1')
        b_conv = bias_variable([chn_base], 'b1')
        conv1_1 = lrelu(conv2d(x1, w_conv, [1, 1]) + b_conv)
        conv1_1_ = lrelu(conv2d(x2, w_conv, [1, 1]) + b_conv)

        w_conv = weight_variable([3, 3, chn_base, chn_base], 'w2')
        b_conv = bias_variable([chn_base], 'b2')
        conv1_2 = lrelu(conv2d(conv1_1, w_conv, [2, 2]) + b_conv)
        conv1_2_ = lrelu(conv2d(conv1_1_, w_conv, [2, 2]) + b_conv)
        # shape = [None, 16, 36, chn_base]

        # Encoder 2
        w_conv = weight_variable([3, 3, chn_base, chn_base * 2], 'w3')
        b_conv = bias_variable([chn_base * 2], 'b3')
        conv2_1 = lrelu(conv2d(conv1_2, w_conv, [1, 1]) + b_conv)
        conv2_1_ = lrelu(conv2d(conv1_2_, w_conv, [1, 1]) + b_conv)

        w_conv = weight_variable([3, 3, chn_base * 2, chn_base * 2], 'w4')
        b_conv = bias_variable([chn_base * 2], 'b4')
        conv2_2 = lrelu(conv2d(conv2_1, w_conv, [2, 2]) + b_conv)
        conv2_2_ = lrelu(conv2d(conv2_1_, w_conv, [2, 2]) + b_conv)
        # shape = [None, 8, 18, chn_base * 2]

        # Encoder 3
        w_conv = weight_variable([3, 3, chn_base * 2, chn_base * 3], 'w5')
        b_conv = bias_variable([chn_base * 3], 'b5')
        conv3_1 = lrelu(conv2d(conv2_2, w_conv, [1, 1]) + b_conv)
        conv3_1_ = lrelu(conv2d(conv2_2_, w_conv, [1, 1]) + b_conv)

        w_conv = weight_variable([3, 3, chn_base * 3, chn_base * 3], 'w6')
        b_conv = bias_variable([chn_base * 3], 'b6')
        conv3_2 = lrelu(conv2d(conv3_1, w_conv, [2, 2]) + b_conv)
        conv3_2_ = lrelu(conv2d(conv3_1_, w_conv, [2, 2]) + b_conv)
        # shape = [None, 4, 9, chn_base * 3]

        # Mapping
        w_conv = weight_variable([3, 3, chn_base * 3, chn_base * 3], 'w7')
        b_conv = bias_variable([chn_base * 3], 'b7')
        conv4_1 = lrelu(conv2d(conv3_2, w_conv, [1, 1]) + b_conv)
        conv4_1_ = lrelu(conv2d(conv3_2_, w_conv, [1, 1]) + b_conv)
    return tf.abs(conv1_1 - conv1_1_), tf.abs(conv2_1 - conv2_1_), tf.abs(conv3_1 - conv3_1_), tf.abs(conv4_1 - conv4_1_)
