#!/usr/bin/env python
# -*- coding: utf-8 -*-
### modified from https://github.com/ppwwyyxx/tensorpack

import os
import sys
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

# https://github.com/apache/arrow/pull/1223#issuecomment-359895666
old_mod = sys.modules.get('torch', None)
sys.modules['torch'] = None
try:
    import pyarrow as pa
except ImportError:
    pa = None
if old_mod is not None:
    sys.modules['torch'] = old_mod
else:
    del sys.modules['torch']

import pickle

__all__ = ['loads', 'dumps', 'dump_pkl', 'load_pkl']


def dumps_msgpack(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return msgpack.dumps(obj, use_bin_type=True)


def loads_msgpack(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return msgpack.loads(buf, raw=False)


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


def dump_pkl(name, obj):
    with open('{}.pkl'.format(name), 'wb') as f:
        pickle.dump( obj, f, pickle.HIGHEST_PROTOCOL )

def load_pkl(name):
    with open('{}.pkl'.format(name), 'rb') as f:
        ret = pickle.load( f )
    return ret

if pa is None:
    loads = loads_msgpack
    dumps = dumps_msgpack
else:
    loads = loads_pyarrow
    dumps = dumps_pyarrow

