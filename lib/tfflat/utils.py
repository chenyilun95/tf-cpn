#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
import numpy as np
from datetime import datetime

def mem_info():
    import subprocess
    dev = subprocess.check_output(
        "nvidia-smi | grep MiB | awk -F '|' '{print $3}' | awk -F '/' '{print $1}' | grep -Eo '[0-9]{1,10}'",
        shell=True)
    dev = dev.decode()
    dev_mem = list(map(lambda x: int(x), dev.split('\n')[:-1]))
    return dev_mem

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

def make_link(dest_path, link_path):
    if os.path.islink(link_path):
        os.system('rm {}'.format(link_path))
    os.system('ln -s {} {}'.format(dest_path, link_path))

def make_dir(path):
    if os.path.exists(path) or os.path.islink(path):
        return
    os.makedirs(path)

def del_file(path, msg='{} deleted.'):
    if os.path.exists(path):
        os.remove(path)
        print(msg.format(path))

def approx_equal(a, b, eps=1e-9):
    return np.fabs(a-b) < eps

def get_rng(obj=None):
    """
    Get a good RNG seeded with time, pid and the object.

    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    return np.random.RandomState(seed)

