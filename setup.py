from setuptools import setup
import os.path as osp

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))

__version__ = '0.2.4'

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []

ext_modules = [
    CUDAExtension('svox.csrc', [
        'svox/csrc/svox.cpp',
        'svox/csrc/svox_kernel.cu',
        'svox/csrc/rt_kernel.cu',
    ], include_dirs=[osp.join(ROOT_DIR, "svox", "csrc", "include")],
    optional=True),
]

setup(
    name='svox',
    version=__version__,
    author='Alex Yu',
    author_email='alexyu99126@gmail.com',
    description='Sparse voxel N^3-tree data structure using CUDA',
    long_description='Sparse voxel N^3-tree data structure PyTorch extension, using CUDA',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    packages=['svox', 'svox.csrc'],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
