#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor pcl_aabb_cuda_forward(
        torch::Tensor pcl_pos,
        torch::Tensor voxel_bound,
        torch::Tensor pcl_bid,
        torch::Tensor voxel_bid);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor pcl_aabb_forward(
        torch::Tensor pcl_pos,
        torch::Tensor voxel_bound,
        torch::Tensor pcl_bid,
        torch::Tensor voxel_bid) {
    
    CHECK_INPUT(pcl_pos);
    CHECK_INPUT(voxel_bound);
    CHECK_INPUT(pcl_bid);
    CHECK_INPUT(voxel_bid);

    return pcl_aabb_cuda_forward(pcl_pos, voxel_bound, pcl_bid, voxel_bid);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &pcl_aabb_forward, "PCL AABB Inside Test (CUDA)");
}
