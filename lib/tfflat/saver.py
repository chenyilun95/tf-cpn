import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import os
import os.path as osp

def get_variables_in_checkpoint_file(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print(
                "It's likely that your checkpoint file has been compressed "
                "with SNAPPY.")

class Saver(object):
    def __init__(self, sess, var_list, model_dump_dir, name_prefix='snapshot'):
        self.sess = sess
        self.var_list = var_list
        self.model_dump_dir = model_dump_dir
        self._name_prefix = name_prefix
        
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)

    def save_model(self, iter):
        filename = '{}_{:d}'.format(self._name_prefix, iter) + '.ckpt'
        if not os.path.exists(self.model_dump_dir):
            os.makedirs(self.model_dump_dir)
        filename = os.path.join(self.model_dump_dir, filename)
        self.saver.save(self.sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

def load_model(sess, model_path):
    #TODO(global variables ?? how about _adam weights)
    variables = tf.global_variables()
    var_keep_dic = get_variables_in_checkpoint_file(model_path)
    if 'global_step' in var_keep_dic:
        var_keep_dic.pop('global_step')

    # vis_var_keep_dic = []
    variables_to_restore = []
    for v in variables:
        if v.name.split(':')[0] in var_keep_dic:
            # print('Varibles restored: %s' % v.name)
            variables_to_restore.append(v)
            # vis_var_keep_dic.append(v.name.split(':')[0])
        else:
            # print('Unrestored Variables: %s' % v.name)
            pass
    # print('Extra Variables in ckpt', set(var_keep_dic) - set(vis_var_keep_dic))

    if len(variables_to_restore) > 0:
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, model_path)
    else:
        print('No variables in {} fits the network'.format(model_path))
