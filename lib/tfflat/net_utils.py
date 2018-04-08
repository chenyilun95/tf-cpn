import tensorflow as tf
from collections import namedtuple

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        if grad_and_vars[0][0] is None:
            print('No gradient on var {}'.format(grad_and_vars[0][1].name))
            continue
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def sum_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    sum_grads = []
    for grad_and_vars in zip(*tower_grads):
        if grad_and_vars[0][0] is None:
            print('No gradient on var {}'.format(grad_and_vars[0][1].name))
            continue
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            if g is not None:
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_sum(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        sum_grads.append(grad_and_var)
    return sum_grads

def aggregate_batch(data_holder):
    results = []
    if len(data_holder) == 1:
        results = data_holder if isinstance(data_holder[0], tf.Tensor) else data_holder[0]
    elif isinstance(data_holder[0], tf.Tensor):
        results.append( tf.concat(data_holder, axis=0) )
    else:
        for i in range(len(data_holder[0])):
            results.append(
                tf.concat([data_holder[j][i] for j in range(len(data_holder))], axis=0))
    return results

def get_optimizer(lr, optimizer='momentum'):
    if optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    elif optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(lr, 0.9)
    elif optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(lr)
    else:
        raise ValueError('invalid optimizer')
    return optimizer

def get_tower_summary_dict(summary):
    ret = dict()
    for v, method in summary:
        if len(tf.get_collection(v)) == 1:
            ret[v] = tf.get_collection(v)[0]
        elif len(tf.get_collection(v)) > 1:
            if method == 'mean':
                ret[v] = tf.reduce_mean(tf.get_collection(v), axis=0)
            elif method == 'sum':
                ret[v] = tf.reduce_sum(tf.get_collection(v), axis=0)
            elif method == 'concat':
                ret[v] = tf.concat(tf.get_collection(v), axis=0)
            else:
                raise ValueError('Invalid summary reduced method: {}'.format(method))
    return ret
