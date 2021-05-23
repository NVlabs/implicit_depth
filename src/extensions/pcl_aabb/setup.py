from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pcl_aabb',
    ext_modules=[
        CUDAExtension('pcl_aabb', [
            'pcl_aabb_cuda.cpp',
            'pcl_aabb_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
