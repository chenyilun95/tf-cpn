# This file implements the nms operator used in network
import megbrain as mgb
from megbrain.craniotome import CraniotomeBase
from megbrain.craniotome import make_opr
from megskull.opr.all import MGBOprForwarderBase, SingleCNOperatorNodeBase, NonTrainableMLPOperatorNodeBase
import numpy as np
from IPython import embed

from lib_kernel.lib_nms.gpu_nms import gpu_nms

class NMSKeepCran(CraniotomeBase):
    __nr_inputs__ = 1
    __nr_outputs__= 1
    __is_dynamic_output_shape__ = True

    def setup(self, iou_threshold):
        self._iou_threshold = iou_threshold

    def execute(self, inputs, outputs):
        """ inputs: list of (x0, y0, x1, y1, score)"""
        in_ = inputs[0].get_value()
        keep = gpu_nms(in_, thresh=self._iou_threshold)
        outputs[0].set_value(keep)

    def grad(self, wrt_idx, inputs, outputs, out_grad):
        return 0

    def init_output_dtype(self, input_dtypes):
        return [np.int32]

class NMSKeep(NonTrainableMLPOperatorNodeBase):

    def __init__(self, name, box, iou_threshold):
        super().__init__(name, box)
        self._iou_threshold = iou_threshold

    def _init_output_mgbvar(self, env):
        var_box = env.get_mgbvar(self._var_input)
        keep = NMSKeepCran.make(var_box, iou_threshold = self._iou_threshold)
        env.set_mgbvar(self._var_output, keep)
