#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> ray_aabb_cuda_forward(
        torch::Tensor ray_dir,
        torch::Tensor voxel_bound,
        torch::Tensor ray_bid,
        torch::Tensor voxel_bid);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> ray_aabb_forward(
        torch::Tensor ray_dir,
        torch::Tensor voxel_bound,
        torch::Tensor ray_bid,
        torch::Tensor voxel_bid) {
    
    CHECK_INPUT(ray_dir);
    CHECK_INPUT(voxel_bound);
    CHECK_INPUT(ray_bid);
    CHECK_INPUT(voxel_bid);

    return ray_aabb_cuda_forward(ray_dir, voxel_bound, ray_bid, voxel_bid);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ray_aabb_forward, "Ray AABB Intersection (CUDA)");
}
