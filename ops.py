import tensorflow as tf
import numpy as np

from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.models import Conv2D
from tensorpack.tfutils.argscope import argscope
from tensorpack.models import MaxPooling
from tensorpack.models import GlobalAvgPooling
from tensorpack.models import BatchNorm
from tensorpack.models import FullyConnected


def gating_op(input_, option, c, _id):
    if option.method_name == 'CAM':
        output = input_
    elif option.method_name == 'ADL':
        output = attention_based_dropout(input_, option)
    elif option.method_name == 'AAE' or option.method_name == 'AAE_SAE':
        output = attention_aware_layer(input_, option, c, _id)
    else:
        return input_
        # raise KeyError("Unavailable method: {}".format(option.method_name))

    return output


def attention_based_dropout(input_, option):
    def _get_importance_map(attention):
        return tf.sigmoid(attention)

    def _get_drop_mask(attention, drop_thr):
        max_val = tf.reduce_max(attention, axis=[1, 2, 3], keepdims=True)
        thr_val = max_val * drop_thr
        return tf.cast(attention < thr_val, dtype=tf.float32, name='drop_mask')

    def _select_component(importance_map, drop_mask, drop_prob):
        random_tensor = tf.random_uniform([], drop_prob, 1. + drop_prob)
        binary_tensor = tf.cast(tf.floor(random_tensor), dtype=tf.float32)
        return (1. -
                binary_tensor) * importance_map + binary_tensor * drop_mask

    ctx = get_current_tower_context()
    is_training = ctx.is_training

    drop_prob = 1 - option.adl_keep_prob
    drop_thr = option.adl_threshold

    if is_training:
        attention_map = tf.reduce_mean(input_, axis=1, keepdims=True)
        importance_map = _get_importance_map(attention_map)
        drop_mask = _get_drop_mask(attention_map, drop_thr)
        selected_map = _select_component(importance_map, drop_mask, drop_prob)
        output = input_ * selected_map
        return output

    else:
        return input_


def convnormrelu(x, name, chan, kernel_size=3, padding='SAME'):
    x = Conv2D(name, x, chan, kernel_size=kernel_size, padding=padding)
    x = tf.nn.relu(x, name=name + '_relu')
    return x


def conv1x3x3x1(x, name, chan, padding='SAME', activation=tf.identity):
    x = Conv2D(name + '1x3',
               x,
               chan,
               kernel_size=(1, 3),
               padding=padding,
               activation=activation)
    x = tf.nn.relu(x, name=name + '1x3_relu')
    x = Conv2D(name + '3x1',
               x,
               chan,
               kernel_size=(3, 1),
               padding=padding,
               activation=activation)
    x = tf.nn.relu(x, name=name + '3x1_relu')
    return x


def conv_bn_relu(x,
                 name,
                 chan,
                 kernel_size=3,
                 padding='SAME',
                 activation=tf.identity):
    x = Conv2D(name,
               x,
               chan,
               kernel_size=kernel_size,
               padding=padding,
               activation=activation)
    x = tf.nn.relu(x, name=name + '_relu')
    return x


# def normalize_atten_maps(atten_maps):
#     atten_shape = atten_maps.size()
#     batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1, )),
#                               dim=-1,
#                               keepdim=True)
#     batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1, )),
#                               dim=-1,
#                               keepdim=True)
#     atten_normed = torch.div(
#         atten_maps.view(atten_shape[0:-2] + (-1, )) - batch_mins,
#         batch_maxs - batch_mins)
#     atten_normed = atten_normed.view(atten_shape)

#     return atten_normed

# def erase_feature_maps(atten_map_normed, feature_maps, threshold):
#     if len(atten_map_normed.size()) > 3:
#         atten_map_normed = torch.squeeze(atten_map_normed)
#     atten_shape = atten_map_normed.size()

#     pos = torch.ge(atten_map_normed, threshold)
#     mask = torch.ones(atten_shape).cuda()
#     mask[pos.data] = 0.0
#     mask = torch.unsqueeze(mask, dim=1)
#     #erase
#     erased_feature_maps = feature_maps * Variable(mask)

#     return erased_feature_maps


def attention_aware_layer(input_, option, c, _id):
    def _get_drop_mask(attention, aae_thr):
        all_val = tf.reduce_sum(attention, axis=[1, 2, 3], keepdims=True)
        avg_val = tf.divide(attention, all_val)
        mask = tf.cast(avg_val < aae_thr, dtype=tf.float32,
                       name='drop_mask')  # < aae_thr ==>1, >aae_thr ===>0
        return mask

    aae_thr = option.aae_threshold
    # aware_l = convnormrelu(input_, "aware", c, kernel_size=1)
    attention_map = tf.reduce_mean(input_, axis=1, keepdims=True)
    mask = _get_drop_mask(attention_map, aae_thr)
    mask_feature = input_ * mask

    # filters = tf.constant(value=1, shape=[3, 3, c, c], dtype=tf.float32)
    # output_ = tf.nn.atrous_conv2d(mask_feature,
    #                               filters,
    #                               2,
    #                               padding="SAME",
    #                               name="atrous")
    with argscope(Conv2D, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(scale=2.)), \
         argscope([Conv2D, MaxPooling, BatchNorm, GlobalAvgPooling],
                  data_format='channels_first'):

        output_ = convnormrelu(mask_feature,
                               "aware_" + str(_id),
                               c,
                               kernel_size=1)
    output = tf.maximum(input_, output_)
    return output