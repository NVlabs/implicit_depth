from torch.utils.cpp_extension import load
ray_aabb = load(
    'ray_aabb', ['extensions/ray_aabb/ray_aabb_cuda.cpp', 'extensions/ray_aabb/ray_aabb_cuda_kernel.cu'], verbose=True)
# help(ray_aabb)
