import tensorflow as tf
import tensorflow.contrib.slim as slim
from . import resnet_v1, resnet_utils
from tensorflow.contrib.slim import arg_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import regularizers, \
    initializers, layers
from config import cfg

def resnet_arg_scope(bn_is_training,
                     bn_trainable,
                     trainable=True,
                     weight_decay=cfg.weight_decay,
                     batch_norm_decay=0.99,
                     batch_norm_epsilon=1e-9,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': bn_is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': bn_trainable,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }

    with arg_scope(
            [slim.conv2d],
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            weights_initializer=initializers.variance_scaling_initializer(),
            trainable=trainable,
            activation_fn=nn_ops.relu,
            normalizer_fn=layers.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

def resnet50(image, bn_is_training, bn_trainable):
    bottleneck = resnet_v1.bottleneck
    blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 1)]),
        resnet_utils.Block('block2', bottleneck,
                           [(512, 128, 2)] + [(512, 128, 1)] * 3),
        resnet_utils.Block('block3', bottleneck,
                           [(1024, 256, 2)] + [(1024, 256, 1)] * 5),
        resnet_utils.Block('block4', bottleneck,
                           [(2048, 512, 2)] + [(2048, 512, 1)] * 2)
    ]
    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):
        with tf.variable_scope('resnet_v1_50', 'resnet_v1_50'):
            net = resnet_utils.conv2d_same(
                image, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')
        net, _ = resnet_v1.resnet_v1(                                  # trainable ?????
            net, blocks[0:1],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_50')

    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):
        net2, _ = resnet_v1.resnet_v1(
            net, blocks[1:2],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_50')
    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):
        net3, _ = resnet_v1.resnet_v1(
            net2, blocks[2:3],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_50')
    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):
        net4, _ = resnet_v1.resnet_v1(
            net3, blocks[3:4],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_50')

    resnet_features = [net, net2, net3, net4]
    return resnet_features

def resnet101(image, bn_is_training, bn_trainable):
    bottleneck = resnet_v1.bottleneck
    blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 1)]),
        resnet_utils.Block('block2', bottleneck,
                           [(512, 128, 2)] + [(512, 128, 1)] * 3),
        resnet_utils.Block('block3', bottleneck,
                           [(1024, 256, 2)] + [(1024, 256, 1)] * 22),
        resnet_utils.Block('block4', bottleneck,
                           [(2048, 512, 2)] + [(2048, 512, 1)] * 2)
    ]
    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):
        with tf.variable_scope('resnet_v1_101', 'resnet_v1_101'):
            net = resnet_utils.conv2d_same(
                image, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')
        net, _ = resnet_v1.resnet_v1(                                  # trainable ?????
            net, blocks[0:1],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_101')

    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):
        net2, _ = resnet_v1.resnet_v1(
            net, blocks[1:2],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_101')
    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):
        net3, _ = resnet_v1.resnet_v1(
            net2, blocks[2:3],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_101')
    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):
        net4, _ = resnet_v1.resnet_v1(
            net3, blocks[3:4],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_101')

    resnet_features = [net, net2, net3, net4]
    return resnet_features

