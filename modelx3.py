import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d

def weight_variable(shape, name=None):
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.001))


def bias_variable(shape, name=None):
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(0))


def conv2d(x, w, strides=1, name=None):
    return tf.nn.conv2d(x, w, strides=[1, 1, strides, 1], padding="SAME", name=name)


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)


def prelu(x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=1,
                                 dtype=x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, x) + _alpha * tf.minimum(0.0, x), _alpha


def deconv(x, w, output_shape, strides, name=None):
    dyn_input_shape = tf.shape(x)
    batch_size = dyn_input_shape[0]
    output_shape = tf.stack([batch_size, output_shape[1], output_shape[2], output_shape[3]])
    output = tf.nn.conv2d_transpose(x, w, output_shape, strides, padding="SAME", name=name)
    return output


def prefilter(k_size, channel_in, channel_out, name=None):
    x = np.linspace(0, 80, num=k_size)
    filters = np.zeros([k_size, 1])
    filters[int((k_size - 1) / 2), 0] = 1
    for chn in range(channel_out - 1):
        y = np.exp(-np.square(x - 40) / (200 / ((channel_out - 1) * 5 + 1) * (chn * 5 + 1)))
        value = interp1d(x, y, kind='cubic')
        value = value(x)
        value = value / np.sum(value)
        filters = np.concatenate((filters, np.expand_dims(value, axis=1)), axis=1)
    filters = np.tile(filters, [1, channel_in, 1, 1])
    filters = np.transpose(filters, (0, 2, 1, 3))
    return tf.get_variable(name=name, shape=[1, k_size, channel_in, channel_out], dtype=tf.float32, initializer=tf.constant_initializer(filters))


def shear(x, scale):
    global y
    input_shape = x.get_shape().as_list()
    hei = input_shape[1]
    wid = input_shape[2]
    shift_max = np.ceil((hei - 1) / 2 * abs(scale))
    base_shift = shift_max - (hei - 1) / 2 * abs(scale)

    paddings = [[0, 0], [0, 0], [int(shift_max), int(shift_max)], [0, 0]]
    x = tf.pad(x, paddings)
    for i in range(hei):
        if scale > 0:
            shift = i * scale + base_shift
        else:
            shift = (hei - i - 1) * abs(scale) + base_shift
        if shift == int(shift):
            cur_y = tf.slice(x, [0, i, int(shift), 0], [-1, 1, wid, -1])
        else:
            cur_y = tf.add((shift - np.floor(shift)) * tf.slice(x, [0, i, int(np.ceil(shift)), 0], [-1, 1, wid, -1]),
                           (np.ceil(shift) - shift) * tf.slice(x, [0, i, int(np.floor(shift)), 0], [-1, 1, wid, -1]))
        if i == 0:
            y = cur_y
        else:
            y = tf.concat([y, cur_y], axis=1)
    return y


def reconstructor(up_scale, x, shear_value=0, chn=27):
    with tf.variable_scope('SR', reuse=tf.AUTO_REUSE):
        input_shape = x.get_shape().as_list()
        size_wid = [int(input_shape[2] / 4), int(input_shape[2] / 2), input_shape[2]]
        ang_in = input_shape[1]
        chn_in = input_shape[3]
        ang_out = (ang_in - 1) * up_scale + 1
        chn_Laplacian = 10
        num_prefilter = 20

        # Shear feature maps
        s0 = shear(x, shear_value)

        """Decomposition"""
        # Layer 1
        w = weight_variable([5, 5, chn_in, chn_Laplacian], 'w1')
        b = bias_variable([chn_Laplacian], 'b1')
        s1 = lrelu(conv2d(s0, w, 4) + b)
        w = weight_variable([3, 3, chn_Laplacian, chn_Laplacian], 'Dw1_1')
        b = bias_variable([chn_Laplacian], 'Db1_1')
        s1_2 = lrelu(deconv(s1, w, [-1, ang_in, size_wid[1], chn_Laplacian], [1, 1, 2, 1]) + b)

        # Layer 2
        w = weight_variable([5, 5, chn_in, chn_Laplacian], 'w2')
        b = bias_variable([chn_Laplacian], 'b2')
        s2 = lrelu(conv2d(s0, w, 2) + b)
        w = weight_variable([5, 5, chn_Laplacian, chn_Laplacian], 'Dw2_1')
        b = bias_variable([chn_Laplacian], 'Db2_1')
        s2_2 = lrelu(deconv(s2, w, [-1, ang_in, size_wid[2], chn_Laplacian], [1, 1, 2, 1]) + b)
        s2 = tf.subtract(s2, s1_2)

        # Layer 3
        w = weight_variable([5, 5, chn_in, chn_Laplacian], 'w3')
        b = bias_variable([chn_Laplacian], 'b3')
        s3 = lrelu(conv2d(s0, w, 1) + b)
        s3 = tf.subtract(s3, s2_2)

        """Pre-filter"""
        w = prefilter(k_size=5, channel_in=chn_Laplacian, channel_out=num_prefilter, name='Prefilter1')
        s1 = conv2d(s1, w, 1)
        w = prefilter(k_size=11, channel_in=chn_Laplacian, channel_out=num_prefilter, name='Prefilter2')
        s2 = conv2d(s2, w, 1)
        w = prefilter(k_size=21, channel_in=chn_Laplacian, channel_out=num_prefilter, name='Prefilter3')
        s3 = conv2d(s3, w, 1)

        """Feature extraction"""
        w = weight_variable([3, 3, num_prefilter, chn], 'w4')
        b = bias_variable([chn], 'b4')
        s1 = lrelu(conv2d(s1, w, 1) + b)

        w = weight_variable([3, 3, num_prefilter, chn], 'w5')
        b = bias_variable([chn], 'b5')
        s2 = lrelu(conv2d(s2, w, 1) + b)

        w = weight_variable([3, 3, num_prefilter, chn], 'w6')
        b = bias_variable([chn], 'b6')
        s3 = lrelu(conv2d(s3, w, 1) + b)

        """Concatenation"""
        w = weight_variable([5, 5, chn, chn], 'Dw3')
        b = bias_variable([chn], 'Db3')
        s1 = lrelu(deconv(s1, w, [-1, ang_in, size_wid[2], chn], [1, 1, 4, 1]) + b)

        w = weight_variable([5, 5, chn, chn], 'Dw4')
        b = bias_variable([chn], 'Db4')
        s2 = lrelu(deconv(s2, w, [-1, ang_in, size_wid[2], chn], [1, 1, 2, 1]) + b)

        s = tf.concat([s1, s2, s3], -1)

        """Mapping"""
        w = weight_variable([3, 3, chn * 3, chn * 3], 'w7')
        b = bias_variable([chn * 3], 'b7')
        s = lrelu(tf.layers.batch_normalization(conv2d(s, w, 1) + b))

        """Angular reconstruction & inverse shear"""
        w = weight_variable([9, 9, chn, chn * 3], 'Dw5')
        b = bias_variable([chn], 'Db5')
        s = deconv(s, w, [-1, ang_out, size_wid[2], chn], [1, up_scale, 1, 1]) + b
        h = shear(s, -shear_value / up_scale)
    return h


def blender(x, chn=27):
    with tf.variable_scope('Blender'):
        input_shape = x.get_shape().as_list()
        size_wid = [int(input_shape[2] / 4), int(input_shape[2] / 2), input_shape[2]]
        chn_in = input_shape[3]
        ang_in = input_shape[1]

        # Blending
        w = weight_variable([1, 1, chn_in, chn], 'w0')
        b = bias_variable([chn], 'b0')
        h0 = lrelu(conv2d(x, w, 1) + b)

        # Encoder: Stride 2
        w = weight_variable([3, 3, chn, chn * 2], 'w1')
        b = bias_variable([chn * 2], 'b1')
        h1 = lrelu(conv2d(h0, w, 2) + b)

        w = weight_variable([3, 3, chn * 2, chn * 2], 'w2')
        b = bias_variable([chn * 2], 'b2')
        h1 = lrelu(conv2d(h1, w, 1) + b)

        # Encoder: Stride 2
        w = weight_variable([3, 3, chn * 2, chn * 2], 'w3')
        b = bias_variable([chn * 2], 'b3')
        h2 = lrelu(conv2d(h1, w, 2) + b)

        w = weight_variable([3, 3, chn * 2, chn * 2], 'w4')
        b = bias_variable([chn * 2], 'b4')
        h2 = lrelu(conv2d(h2, w, 1) + b)

        # Mapping
        w = weight_variable([3, 3, chn * 2, chn * 2], 'w5')
        b = bias_variable([chn * 2], 'b5')
        h2 = lrelu(tf.layers.batch_normalization(conv2d(h2, w, 1) + b))

        # Decoder: Stride 2
        w = weight_variable([5, 5, chn * 2, chn * 2], 'Dw1')
        b = bias_variable([chn * 2], 'Db1')
        h2 = deconv(h2, w, [-1, ang_in, size_wid[1], chn * 2], [1, 1, 2, 1]) + b

        h3 = tf.concat([lrelu(h2), h1], 3)

        w = weight_variable([1, 1, chn * 4, chn * 2], 'Dw2')
        b = bias_variable([chn * 2], 'Db2')
        h3 = lrelu(conv2d(h3, w, 1) + b)

        # Decoder: Stride 2
        w = weight_variable([5, 5, chn, chn * 2], 'Dw3')
        b = bias_variable([chn], 'Db3')
        h4 = deconv(h3, w, [-1, ang_in, size_wid[2], chn], [1, 1, 2, 1]) + b

        h4 = tf.concat([lrelu(h4), h0], 3)

        w = weight_variable([1, 1, chn * 2, chn], 'Dw4')
        b = bias_variable([chn], 'Db4')
        h4 = lrelu(conv2d(h4, w, 1) + b)

        w = weight_variable([9, 9, chn, 1], 'w6')  # The difference with old model
        b = bias_variable([1], 'b6')
        h = conv2d(h4, w, 1) + b
    return h


def model(up_scale, x):
    input_shape = x.get_shape().as_list()
    size_wid = [int(input_shape[2] / 4), int(input_shape[2] / 2), input_shape[2]]
    ang_in = input_shape[1]
    ang_out = (ang_in - 1) * up_scale + 1
    chn_base = 27

    # Shear reconstructor
    s1 = reconstructor(up_scale, x, shear_value=-9, chn=chn_base)
    s2 = reconstructor(up_scale, x, shear_value=-6, chn=chn_base)
    s3 = reconstructor(up_scale, x, shear_value=-3, chn=chn_base)
    s4 = reconstructor(up_scale, x, shear_value=0, chn=chn_base)
    s5 = reconstructor(up_scale, x, shear_value=3, chn=chn_base)
    s6 = reconstructor(up_scale, x, shear_value=6, chn=chn_base)
    s7 = reconstructor(up_scale, x, shear_value=9, chn=chn_base)

    s = tf.concat([s1, s2, s3, s4, s5, s6, s7], axis=-1)

    # Shear blender
    y_out = blender(s, chn=chn_base)
    return y_out
