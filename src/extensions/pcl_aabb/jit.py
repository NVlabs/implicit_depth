from torch.utils.cpp_extension import load
pcl_aabb = load(
    'pcl_aabb', ['extensions/pcl_aabb/pcl_aabb_cuda.cpp', 'extensions/pcl_aabb/pcl_aabb_cuda_kernel.cu'], verbose=True)
# help(pcl_aabb)
