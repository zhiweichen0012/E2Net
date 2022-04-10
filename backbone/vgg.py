import tensorflow as tf

from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.models import Conv2D
from tensorpack.models import MaxPooling
from tensorpack.models import GlobalAvgPooling
from tensorpack.models import BatchNorm
from tensorpack.models import FullyConnected

from ops import gating_op
from ops import convnormrelu
from ops import conv1x3x3x1

__all__ = ['vgg_gap']


@auto_reuse_variable_scope
def vgg_gap(image, option):
    with argscope(Conv2D, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(scale=2.)), \
         argscope([Conv2D, MaxPooling, BatchNorm, GlobalAvgPooling],
                  data_format='channels_first'):

        l = convnormrelu(image, 'conv1_1', 64)
        if option.gating_position[11]: l = gating_op(l, option, 64, 11)
        l = convnormrelu(l, 'conv1_2', 64)
        if option.gating_position[12]: l = gating_op(l, option, 64, 12)
        l = MaxPooling('pool1', l, 2)
        if option.gating_position[1]: l = gating_op(l, option, 64, 1)
        p1 = l

        l = convnormrelu(l, 'conv2_1', 128)
        if option.gating_position[21]: l = gating_op(l, option, 128, 21)
        l = convnormrelu(l, 'conv2_2', 128)
        if option.gating_position[22]: l = gating_op(l, option, 128, 22)
        l = MaxPooling('pool2', l, 2)
        if option.gating_position[2]: l = gating_op(l, option, 128, 2)
        p2 = l

        l = convnormrelu(l, 'conv3_1', 256)
        if option.gating_position[31]: l = gating_op(l, option, 256, 31)
        l = convnormrelu(l, 'conv3_2', 256)
        if option.gating_position[32]: l = gating_op(l, option, 256, 32)
        l = convnormrelu(l, 'conv3_3', 256)
        if option.gating_position[33]: l = gating_op(l, option, 256, 33)
        l = MaxPooling('pool3', l, 2)
        if option.gating_position[3]: l = gating_op(l, option, 256, 3)
        p3 = l

        l = convnormrelu(l, 'conv4_1', 512)
        if option.gating_position[41]: l = gating_op(l, option, 512, 41)
        l = convnormrelu(l, 'conv4_2', 512)
        if option.gating_position[42]: l = gating_op(l, option, 512, 42)
        l = convnormrelu(l, 'conv4_3', 512)
        if option.gating_position[43]: l = gating_op(l, option, 512, 43)
        l = MaxPooling('pool4', l, 2)
        if option.gating_position[4]: l = gating_op(l, option, 512, 4)
        p4 = l

        l = convnormrelu(l, 'conv5_1', 512)
        if option.gating_position[51]: l = gating_op(l, option, 512, 51)
        l = convnormrelu(l, 'conv5_2', 512)
        if option.gating_position[52]: l = gating_op(l, option, 512, 52)

        l = convnormrelu(l, 'conv5_3', 512)
        # l = conv1x3x3x1(l, 'conv6', 512, activation=get_bn())
        if option.gating_position[53]: l = gating_op(l, option, 512, 53)
        c53 = l

        if (option.method_name == "SAE"
                or option.method_name == 'AAE_SAE') and False:
            l = conv1x3x3x1(l, 'conv6', 512, activation=get_bn())
            if option.gating_position[54]: l = gating_op(l, option, 512, 54)

        convmaps = convnormrelu(l, 'new', 1024)
        if option.gating_position[6]: convmaps = gating_op(l, option, 1024, 6)

        p_logits = GlobalAvgPooling('gap', convmaps)
        logits = FullyConnected(
            'linear',
            p_logits,
            option.number_of_class,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        # 1x3 branch
        l_1x3 = convnormrelu(c53, 'conv1x3', 512, kernel_size=(1, 3))
        convmaps_1x3 = convnormrelu(l_1x3, 'new1x3', 1024)
        p_logits_1x3 = GlobalAvgPooling('gap1x3', convmaps_1x3)
        logits_1x3 = FullyConnected(
            '1x3_linear',
            p_logits_1x3,
            option.number_of_class,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        # 3x1 branch
        l_3x1 = convnormrelu(c53, 'conv3x1', 512, kernel_size=(3, 1))
        convmaps_3x1 = convnormrelu(l_3x1, 'new3x1', 1024)
        p_logits_3x1 = GlobalAvgPooling('gap3x1', convmaps_3x1)
        logits_3x1 = FullyConnected(
            '3x1_linear',
            p_logits_3x1,
            option.number_of_class,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01))

    return [logits, logits_1x3,
            logits_3x1], [convmaps, convmaps_1x3, convmaps_3x1]


def get_bn(zero_init=False):
    if zero_init:
        return lambda x, name=None: BatchNorm(
            'bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)