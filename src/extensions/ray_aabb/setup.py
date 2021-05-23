from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ray_aabb',
    ext_modules=[
        CUDAExtension('ray_aabb', [
            'ray_aabb_cuda.cpp',
            'ray_aabb_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
