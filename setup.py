from setuptools import setup
import os.path as osp

ROOT_DIR = osp.dirname(osp.abspath(__file__))

__version__ = '0.1.2'

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []

setup(
    name='svox',
    version=__version__,
    author='Alex Yu',
    author_email='alexyu99126@gmail.com',
    description='Sparse voxel N^3-tree data structure using numpy',
    long_description='Sparse voxel N^3-tree data structure using numpy',
    packages=['svox'],
)
