# copy from tensorpack: https://github.com/ppwwyyxx/tensorpack
import numpy as np
import threading
import multiprocessing as mp
import weakref
from contextlib import contextmanager
from .serialize import loads, dumps
import errno
import uuid
import os
import zmq
import atexit
from itertools import cycle
from copy import copy
from .utils import get_rng
from setproctitle import setproctitle

def del_weakref(x):
    o = x()
    if o is not None:
        o.__del__()

@contextmanager
def _zmq_catch_error(name):
    try:
        yield
    except zmq.ContextTerminated:
        print("[{}] Context terminated.".format(name))
        raise Exception
    except zmq.ZMQError as e:
        if e.errno == errno.ENOTSOCK:       # socket closed
            print("[{}] Socket closed.".format(name))
            raise Exception
        else:
            raise
    except Exception:
        raise

class DataFlowReentrantGuard(object):
    """
    A tool to enforce non-reentrancy.
    Mostly used on DataFlow whose :meth:`get_data` is stateful,
    so that multiple instances of the iterator cannot co-exist.
    """
    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        self._succ = self._lock.acquire(False)
        if not self._succ:
            raise threading.ThreadError("This DataFlow is not reentrant!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
        return False

class DataFromList(object):
    def __init__(self, datalist, is_train=True, shuffle=True):
        self.rng = get_rng()
        self._datalist = datalist
        self._shuffle = shuffle
        self._is_train = is_train

    def get_data(self):
        if self._is_train:
            while True:
                idxs = np.arange(len(self._datalist))
                if self._shuffle:
                    self.rng.shuffle(idxs)
                for i in idxs:
                    yield self._datalist[i]
        else:
            idxs = np.arange(len(self._datalist))
            if self._shuffle:
                self.rng.shuffle(idxs)
            for i in idxs:
                yield self._datalist[i]

    def reset_state(self):
        self.rng = get_rng()

class _ParallelMapData(object):
    def __init__(self, ds, buffer_size):
        assert buffer_size > 0, buffer_size
        self._buffer_size = buffer_size
        self._buffer_occupancy = 0  # actual #elements in buffer

        self.ds = ds

    def _recv(self):
        pass

    def _send(self, dp):
        pass

    def _recv_filter_none(self):
        ret = self._recv()
        assert ret is not None, \
            "[{}] Map function cannot return None when strict mode is used.".format(type(self).__name__)
        return ret

    def _fill_buffer(self, cnt=None):
        if cnt is None:
            cnt = self._buffer_size - self._buffer_occupancy
        try:
            for _ in range(cnt):
                dp = next(self._iter)
                self._send(dp)
        except StopIteration:
            print(
                "[{}] buffer_size cannot be larger than the size of the DataFlow!".format(type(self).__name__))
            raise
        self._buffer_occupancy += cnt

    def get_data_non_strict(self):
        for dp in self._iter:
            self._send(dp)
            ret = self._recv()
            if ret is not None:
                yield ret

        self._iter = self.ds.get_data()   # refresh
        for _ in range(self._buffer_size):
            self._send(next(self._iter))
            ret = self._recv()
            if ret is not None:
                yield ret

    def get_data_strict(self):
        self._fill_buffer()
        for dp in self._iter:
            self._send(dp)
            yield self._recv_filter_none()
        self._iter = self.ds.get_data()   # refresh

        # first clear the buffer, then fill
        for k in range(self._buffer_size):
            dp = self._recv_filter_none()
            self._buffer_occupancy -= 1
            if k == self._buffer_size - 1:
                self._fill_buffer()
            yield dp

class MapData(object):
    """
    Apply a mapper/filter on the DataFlow.

    Note:
        1. Please make sure func doesn't modify the components
           unless you're certain it's safe.
        2. If you discard some datapoints, ``ds.size()`` will be incorrect.
    """

    def __init__(self, ds, func):
        """
        Args:
            ds (DataFlow): input DataFlow
            func (datapoint -> datapoint | None): takes a datapoint and returns a new
                datapoint. Return None to discard this datapoint.
        """
        self.ds = ds
        self.func = func

    def get_data(self):
        for dp in self.ds.get_data():
            ret = self.func(copy(dp))  # shallow copy the list
            if ret is not None:
                yield ret

    def reset_state(self):
        pass

class MultiProcessMapDataZMQ(_ParallelMapData):
    """
    Same as :class:`MapData`, but start processes to run the mapping function,
    and communicate with ZeroMQ pipe.

    Note:
        1. Processes run in parallel and can take different time to run the
           mapping function. Therefore the order of datapoints won't be
           preserved, and datapoints from one pass of `df.get_data()` might get
           mixed with datapoints from the next pass.

           You can use **strict mode**, where `MultiProcessMapData.get_data()`
           is guranteed to produce the exact set which `df.get_data()`
           produces. Although the order of data still isn't preserved.
    """
    class _Worker(mp.Process):
        def __init__(self, identity, map_func, pipename, hwm):
            super(MultiProcessMapDataZMQ._Worker, self).__init__()
            self.identity = identity
            self.map_func = map_func
            self.pipename = pipename
            self.hwm = hwm

        def run(self):
            print('Start data provider {}-{}'.format(self.pipename, self.identity))
            setproctitle('data provider {}-{}'.format(self.pipename, self.identity))
            ctx = zmq.Context()
            socket = ctx.socket(zmq.DEALER)
            socket.setsockopt(zmq.IDENTITY, self.identity)
            socket.set_hwm(self.hwm)
            socket.connect(self.pipename)

            while True:
                dp = loads(socket.recv(copy=False).bytes)
                dp = self.map_func(dp)
                socket.send(dumps(dp), copy=False)

    def __init__(self, ds, nr_proc, map_func, buffer_size=200, strict=False):
        """
        Args:
            ds (DataFlow): the dataflow to map
            nr_proc(int): number of threads to use
            map_func (callable): datapoint -> datapoint | None
            buffer_size (int): number of datapoints in the buffer
            strict (bool): use "strict mode", see notes above.
        """
        _ParallelMapData.__init__(self, ds, buffer_size)
        self.nr_proc = nr_proc
        self.map_func = map_func
        self._strict = strict
        self._procs = []
        self._guard = DataFlowReentrantGuard()

        self._reset_done = False
        self._procs = []

    def _reset_once(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.set_hwm(self._buffer_size * 2)
        pipename = "ipc://@{}-pipe-{}".format('dataflow-map', str(uuid.uuid1())[:8])

        try:
            self.socket.bind(pipename)
        except zmq.ZMQError:
            print(
                "ZMQError in socket.bind(). Perhaps you're \
                using pipes on a non-local file system. See documentation of PrefetchDataZMQ for more information.")
            raise

        self._proc_ids = [u'{}'.format(k).encode('utf-8') for k in range(self.nr_proc)]
        worker_hwm = int(self._buffer_size * 2 // self.nr_proc)
        self._procs = [MultiProcessMapDataZMQ._Worker(
            self._proc_ids[k], self.map_func, pipename, worker_hwm)
            for k in range(self.nr_proc)]

        self.ds.reset_state()
        self._iter = self.ds.get_data()
        self._iter_worker = cycle(iter(self._proc_ids))

        for p in self._procs:
            p.deamon = True
            p.start()
        self._fill_buffer()     # pre-fill the bufer

    def reset_state(self):
        if self._reset_done:
            return
        self._reset_done = True

        # __del__ not guranteed to get called at exit
        atexit.register(del_weakref, weakref.ref(self))

        self._reset_once()  # build processes

    def _send(self, dp):
        # round-robin assignment
        worker = next(self._iter_worker)
        msg = [worker, dumps(dp)]
        self.socket.send_multipart(msg, copy=False)

    def _recv(self):
        msg = self.socket.recv_multipart(copy=False)
        dp = loads(msg[1].bytes)
        return dp

    def get_data(self):
        with self._guard, _zmq_catch_error('MultiProcessMapData'):
            if self._strict:
                for dp in self.get_data_strict():
                    yield dp
            else:
                for dp in self.get_data_non_strict():
                    yield dp

    def __del__(self):
        try:
            if not self._reset_done:
                return
            if not self.context.closed:
                self.socket.close(0)
                self.context.destroy(0)
            for x in self._procs:
                x.terminate()
                x.join(5)
            print("{} successfully cleaned-up.".format(type(self).__name__))
        except Exception:
            pass

class BatchData(object):
    """
    Stack datapoints into batches.
    It produces datapoints of the same number of components as ``ds``, but
    each component has one new extra dimension of size ``batch_size``.
    The batch can be either a list of original components, or (by default)
    a numpy array of original components.
    """

    def __init__(self, ds, batch_size, use_list=False):
        """
        Args:
            ds (DataFlow): When ``use_list=False``, the components of ``ds``
                must be either scalars or :class:`np.ndarray`, and have to be consistent in shapes.
            batch_size(int): batch size
            use_list (bool): if True, each component will contain a list
                of datapoints instead of an numpy array of an extra dimension.
        """
        self.ds = ds
        self.batch_size = int(batch_size)
        self.use_list = use_list

    def get_data(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """
        holder = []
        for data in self.ds.get_data():
            holder.append(data)
            if len(holder) == self.batch_size:
                yield BatchData._aggregate_batch(holder, self.use_list)
                del holder[:]

    @staticmethod
    def _aggregate_batch(data_holder, use_list=False):
        size = len(data_holder[0])
        result = []
        for k in range(size):
            if use_list:
                result.append(
                    [x[k] for x in data_holder])
            else:
                dt = data_holder[0][k]
                if type(dt) in [int, bool]:
                    tp = 'int32'
                elif type(dt) == float:
                    tp = 'float32'
                else:
                    try:
                        tp = dt.dtype
                    except AttributeError:
                        raise TypeError("Unsupported type to batch: {}".format(type(dt)))
                try:
                    result.append(
                        np.asarray([x[k] for x in data_holder], dtype=tp))
                except Exception as e:  # noqa
                    logger.exception("Cannot batch data. Perhaps they are of inconsistent shape?")
                    if isinstance(dt, np.ndarray):
                        s = pprint.pformat([x[k].shape for x in data_holder])
                        logger.error("Shape of all arrays to be batched: " + s)
                    try:
                        # open an ipython shell if possible
                        import IPython as IP; IP.embed()    # noqa
                    except ImportError:
                        pass
        return result

    def reset_state(self):
        self.ds.reset_state()

