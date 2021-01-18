from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = '0.0.1'

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []

ext_modules = [
    CUDAExtension('svox.csrc', [
        'svox/csrc/svox.cpp',
        'svox/csrc/svox_kernel.cu',
    ]),
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
