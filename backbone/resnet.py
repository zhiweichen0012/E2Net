import tensorflow as tf
import sys

from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import Conv2D
from tensorpack.models import MaxPooling
from tensorpack.models import GlobalAvgPooling
from tensorpack.models import BatchNorm
from tensorpack.models import BNReLU
from tensorpack.models import FullyConnected
from ops import convnormrelu

from ops import gating_op
from ops import conv1x3x3x1
from ops import conv_bn_relu

__all__ = ['resnet50_se']

adl_index = {64: 10, 128: 20, 256: 30, 512: 40}


def is_data_format_nchw():
    data_format = get_arg_scope()['Conv2D']['data_format']
    return data_format in ['NCHW', 'channels_first']


def resnet(input_, option):
    mode = option.mode
    DEPTH = option.depth
    bottleneck = {'se': se_resnet_bottleneck}[mode]

    cfg = {
        50: ([3, 4, 6, 3], bottleneck),
    }
    defs, block_func = cfg[DEPTH]
    group_func = resnet_group

    with argscope(Conv2D, use_bias=False, kernel_initializer= \
            tf.variance_scaling_initializer(scale=2.0, mode='fan_out')), \
         argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm],
                  data_format='channels_first'):

        l = Conv2D('conv0', input_, 64, 7, strides=2, activation=BNReLU)
        if option.gating_position[0]: l = gating_op(l, option, 64, 0)

        l = MaxPooling('pool0', l, 3, strides=2, padding='SAME')
        if option.gating_position[1]: l = gating_op(l, option, 64, 1)

        l = group_func('group0', l, block_func, 64, defs[0], 1, option)
        if option.gating_position[2]: l = gating_op(l, option, 64, 2)

        l = group_func('group1', l, block_func, 128, defs[1], 2, option)
        if option.gating_position[3]: l = gating_op(l, option, 128, 3)

        l = group_func('group2', l, block_func, 256, defs[2], 2, option)
        if option.gating_position[4]: l = gating_op(l, option, 256, 4)

        l = group_func('group3', l, block_func, 512, defs[3], 1, option)
        if option.gating_position[5]:
            l = gating_op(l, option, 2048, 5)  #(?, 2048, 14, 14)

        p_logits = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linearnew', p_logits, option.number_of_class)

        # 1x3 branch
        if option.use_bn == 0:
            l_1x3 = convnormrelu(l, 'conv1x3', 2048, kernel_size=(1, 3))
        else:
            l_1x3 = conv_bn_relu(l,
                                 'conv1x3',
                                 2048,
                                 kernel_size=(1, 3),
                                 activation=get_bn())
        p_logits_1x3 = GlobalAvgPooling('gap_1x3', l_1x3)
        logits_1x3 = FullyConnected('linearnew_1x3', p_logits_1x3,
                                    option.number_of_class)
        # 3x1 branch
        if option.use_bn == 0:
            l_3x1 = convnormrelu(l, 'conv3x1', 2048, kernel_size=(3, 1))
        else:
            l_3x1 = conv_bn_relu(l,
                                 'conv3x1',
                                 2048,
                                 kernel_size=(3, 1),
                                 activation=get_bn())
        p_logits_3x1 = GlobalAvgPooling('gap_3x1', l_3x1)
        logits_3x1 = FullyConnected('linearnew_3x1', p_logits_3x1,
                                    option.number_of_class)

    # return logits, l
    return [logits, logits_1x3, logits_3x1], [l, l_1x3, l_3x1]


def resnet_group(name, l, block_func, features, count, stride, option):
    k = adl_index[features]

    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                k = k + 1
                l = block_func(option,
                               l,
                               features,
                               stride if i == 0 else 1,
                               adl_index=k)

    return l


def se_resnet_bottleneck(option, l, ch_out, stride, adl_index=None):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1',
                             squeeze,
                             ch_out // 4,
                             activation=tf.nn.relu)
    squeeze = FullyConnected('fc2',
                             squeeze,
                             ch_out * 4,
                             activation=tf.nn.sigmoid)
    ch_ax = 1 if is_data_format_nchw() else 3
    shape = [-1, 1, 1, 1]
    shape[ch_ax] = ch_out * 4
    l = l * tf.reshape(squeeze, shape)

    out = l + resnet_shortcut(
        shortcut, ch_out * 4, stride, activation=get_bn())
    out = tf.nn.relu(out)

    if option.gating_position[adl_index]:
        out = gating_op(out, option, ch_out * 4, adl_index)
    return out


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    n_in = l.get_shape().as_list()[1 if is_data_format_nchw() else 3]
    if n_in != n_out:
        return Conv2D('convshortcut',
                      l,
                      n_out,
                      1,
                      stride=stride,
                      activation=activation)
    else:
        return l


def get_bn(zero_init=False):
    if zero_init:
        return lambda x, name=None: BatchNorm(
            'bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)


def resnet50_se(input_, option):
    return resnet(input_, option)
