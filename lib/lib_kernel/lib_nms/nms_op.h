/*
@author: zeming li
@contact: zengarden2009@gmail.com
*/

#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_NMS_OP_H_
#define TENSORFLOW_USER_OPS_NMS_OP_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"


#define DIVUP(m,n) (((m) - 1) / (n) + 1)

namespace tensorflow {

// keepout and numout are the kernel output
/*
int* keep_out, int* num_out, 
*/
void NMSForward(const float* boxes_host,
                const float nms_overlap_thresh,
                const int max_out,
                const Eigen::GpuDevice& d);
}

#endif  // TENSORFLOW_CORE_KERNELS_NMS_OP_H_
