import zmq
import multiprocessing as mp
from .serialize import loads, dumps

def data_sender(id, name, func_iter, *args):
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.connect('ipc://@{}'.format(name))

    print('start data provider {}-{}'.format(name, id))
    while True:
        data_iter = func_iter(id, *args)
        for msg in data_iter:
            # print(id)
            sender.send( dumps([id, msg]) )

def provider(nr_proc, name, func_iter, *args):
    proc_ids = [i for i in range(nr_proc)]

    procs = []
    for i in range(nr_proc):
        w = mp.Process(target=data_sender, args=(proc_ids[i], name, func_iter, *args))
        w.deamon = True
        procs.append( w )

    for p in procs:
        p.start()

def receiver(name):
    context = zmq.Context()

    receiver = context.socket(zmq.PULL)
    receiver.bind('ipc://@{}'.format(name))

    while True:
        id, msg = loads( receiver.recv() )
        # print(id, end='')
        yield msg
