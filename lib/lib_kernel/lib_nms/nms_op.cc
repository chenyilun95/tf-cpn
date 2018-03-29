/*
@author: zeming li
@contact: zengarden2009@gmail.com
*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

REGISTER_OP("NMS")
.Attr("T: {float, double}")
.Input("boxes_host: T")
.Input("nms_overlap_thresh: float")
.Input("max_out: int")
.Output("keep_out: int");

using namespace tensorflow;

// class NMSOp : public OpKernel {
// public:
// 	explicit NMSOp(OpKernelConstruction* context) : OpKernel(context) {}
// 	void Compute(OpKernelContext* context) override {
// 		const Tensor& input_tensor = context->input(0);
// 		auto input = input_tensor.flat<float>();
// 		Tensor* output_tensor = NULL;
// 		OP_REQUIRES_OK(context, context->allocate_output(
// 			0, input_tensor.shape(), &output_tensor));
// 	}
// }

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

bool NMSKernel(const float *in, const int N, float* out);

template <class T>
class NMSOpGPU<Eigen::GpuDevice, T>: public OpKernel {
public:
    typedef Eigen::GpuDevice Device;
    explicit NMSOpGPU(OpKernelConstruction* context) : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {
        // Get input tensor
        const Tensor& det = context->input(0);
        const float nms_overlap_thresh = context->input(1);
        const float max_out = context->input(2);
        int num_box = det.dim_size(0);
        int box_dim = det.dim_size(1);

        OP_REQUIRES(context, det.dims() == 2,
            errors::InvalidArgument("det must be 2-dimensional"));

        //create output tensor
        Tensor* keep_out = nullptr;
        int dim;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, det.shape(), &keep_out)); 

        if (!context->status().ok()) {
            return;
        }

        //compute
        dim3 blocks(DIVUP(num_bbox, threadsPerBlock),
              DIVUP(num_bbox, threadsPerBlock));
        dim3 threads(threadsPerBlock);
        
        nms_kernel<<<blocks, threads>>>(num_box,
            nms_overlap_thresh,
            det->flat<float>().data(),
            mask->flat<unsigned long long>().data());

        // nms_kernel<<<blocks, threads>>>(keep_out->flat<int>().data(), 
        //     det->flat<float>().data(), 
        //     num_box, box_dim, nms_overlap_thresh, mask_dev, 
        //     context->eigen_device<Eigen::GpuDevice>());


        //auto input = input_tensor.flat<float>();
        // Create an output tensor
        auto output = output_tensor->flat<float>();
        const int N = input.size();
        NMS(context, &det, &score, )
        if (!cuda_op_launcher(input.data(), N, output.data())){
            context->CtxFailureWithWarning(errors::Internal("FATAL CUDA ERROR"));
            return;
        }
    }
};


REGISTER_KERNEL_BUILDER(Name("NMSOp").Device(DEVICE_GPU), NMSOpGpu);
#endif
REGISTER_KERNEL_BUILDER(Name("NMSOp").Device(DEVICE_CPU), NMSOp);
