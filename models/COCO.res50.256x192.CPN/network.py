import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys, os
import argparse
import numpy as np
from functools import partial

from config import cfg
from tfflat.base import ModelDesc, Trainer
from tfflat.utils import mem_info

from nets.basemodel import resnet50, resnet_arg_scope, resnet_v1
resnet_arg_scope = partial(resnet_arg_scope, bn_trainable=cfg.bn_train)

def create_global_net(blocks, is_training, trainable=True):
    global_fms = []
    global_outs = []
    last_fm = None
    initializer = tf.contrib.layers.xavier_initializer()
    for i, block in enumerate(reversed(blocks)):
        with slim.arg_scope(resnet_arg_scope(bn_is_training=is_training)):
            lateral = slim.conv2d(block, 256, [1, 1],
                trainable=trainable, weights_initializer=initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='lateral/res{}'.format(5-i))

        if last_fm is not None:
            sz = tf.shape(lateral)
            upsample = tf.image.resize_bilinear(last_fm, (sz[1], sz[2]),
                name='upsample/res{}'.format(5-i))
            upsample = slim.conv2d(upsample, 256, [1, 1],
                trainable=trainable, weights_initializer=initializer,
                padding='SAME', activation_fn=None,
                scope='merge/res{}'.format(5-i))
            last_fm = upsample + lateral
        else:
            last_fm = lateral

        with slim.arg_scope(resnet_arg_scope(bn_is_training=is_training)):
            tmp = slim.conv2d(last_fm, 256, [1, 1],
                trainable=trainable, weights_initializer=initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='tmp/res{}'.format(5-i))
            out = slim.conv2d(tmp, cfg.nr_skeleton, [3, 3],
                trainable=trainable, weights_initializer=initializer,
                padding='SAME', activation_fn=None,
                scope='pyramid/res{}'.format(5-i))
        global_fms.append(last_fm)
        global_outs.append(tf.image.resize_bilinear(out, (cfg.output_shape[0], cfg.output_shape[1])))
    global_fms.reverse()
    global_outs.reverse()
    return global_fms, global_outs

def create_refine_net(blocks, is_training, trainable=True):
    initializer = tf.contrib.layers.xavier_initializer()
    bottleneck = resnet_v1.bottleneck
    refine_fms = []
    for i, block in enumerate(blocks):
        mid_fm = block
        with slim.arg_scope(resnet_arg_scope(bn_is_training=is_training)):
            for j in range(i):
                mid_fm = bottleneck(mid_fm, 256, 128, stride=1, scope='res{}/refine_conv{}'.format(2+i, j)) # no projection
        mid_fm = tf.image.resize_bilinear(mid_fm, (cfg.output_shape[0], cfg.output_shape[1]),
            name='upsample_conv/res{}'.format(2+i))
        refine_fms.append(mid_fm)
    refine_fm = tf.concat(refine_fms, axis=3)
    with slim.arg_scope(resnet_arg_scope(bn_is_training=is_training)):
        refine_fm = bottleneck(refine_fm, 256, 128, stride=1, scope='final_bottleneck')
        res = slim.conv2d(refine_fm, cfg.nr_skeleton, [3, 3],
            trainable=trainable, weights_initializer=initializer,
            padding='SAME', activation_fn=None,
            scope='refine_out')
    return res

class Network(ModelDesc):
    def make_data(self):
        from COCOAllJoints import COCOJoints
        from dataset import Preprocessing

        d = COCOJoints()
        train_data, _ = d.load_data(cfg.min_kps)

        from tfflat.data_provider import DataFromList, MultiProcessMapDataZMQ, BatchData, MapData
        dp = DataFromList(train_data)
        if cfg.dpflow_enable:
            dp = MultiProcessMapDataZMQ(dp, cfg.nr_dpflows, Preprocessing)
        else:
            dp = MapData(dp, Preprocessing)
        dp = BatchData(dp, cfg.batch_size // cfg.nr_aug)
        dp.reset_state()
        dataiter = dp.get_data()

        return dataiter

    def make_network(self, is_train):
        if is_train:
            image = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.data_shape, 3])
            label15 = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            label11 = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            label9 = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            label7 = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            valids = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.nr_skeleton])
            labels = [label15, label11, label9, label7]
            # labels.reverse() # The original labels are reversed. For reproduction of our pre-trained model, I'll keep it same.
            self.set_inputs(image, label15, label11, label9, label7, valids)
        else:
            image = tf.placeholder(tf.float32, shape=[None, *cfg.data_shape, 3])
            self.set_inputs(image)

        resnet_fms = resnet50(image, is_train, bn_trainable=True)
        global_fms, global_outs = create_global_net(resnet_fms, is_train)
        refine_out = create_refine_net(global_fms, is_train)

        # make loss
        if is_train:
            def ohkm(loss, top_k):
                ohkm_loss = 0.
                for i in range(cfg.batch_size):
                    sub_loss = loss[i]
                    topk_val, topk_idx = tf.nn.top_k(sub_loss, k=top_k, sorted=False, name='ohkm{}'.format(i))
                    tmp_loss = tf.gather(sub_loss, topk_idx, name='ohkm_loss{}'.format(i)) # can be ignore ???
                    ohkm_loss += tf.reduce_sum(tmp_loss) / top_k
                ohkm_loss /= cfg.batch_size
                return ohkm_loss

            global_loss = 0.
            for i, (global_out, label) in enumerate(zip(global_outs, labels)):
                global_label = label * tf.to_float(tf.greater(tf.reshape(valids, (-1, 1, 1, cfg.nr_skeleton)), 1.1))
                global_loss += tf.reduce_mean(tf.square(global_out - global_label)) / len(labels)
            global_loss /= 2.
            self.add_tower_summary('global_loss', global_loss)
            refine_loss = tf.reduce_mean(tf.square(refine_out - label7), (1,2)) * tf.to_float((tf.greater(valids, 0.1)))
            refine_loss = ohkm(refine_loss, 8)
            self.add_tower_summary('refine_loss', refine_loss)

            total_loss = refine_loss + global_loss
            self.add_tower_summary('loss', total_loss)
            self.set_loss(total_loss)
        else:
            self.set_outputs(refine_out)

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', '-d', type=str, dest='gpu_ids')
        parser.add_argument('--continue', '-c', dest='continue_train', action='store_true')
        parser.add_argument('--debug', dest='debug', action='store_true')
        args = parser.parse_args()

        if not args.gpu_ids:
            args.gpu_ids = str(np.argmin(mem_info()))

        if '-' in args.gpu_ids:
            gpus = args.gpu_ids.split('-')
            gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
            gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
            args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

        return args
    args = parse_args()

    cfg.set_args(args.gpu_ids, args.continue_train)
    trainer = Trainer(Network(), cfg)
    trainer.train()

