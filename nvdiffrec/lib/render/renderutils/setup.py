from setuptools import setup
import torch
import os,glob
from torch.utils.cpp_extension import (CUDAExtension, CppExtension, BuildExtension)

def get_extensions():
    extensions = []
    ext_name = 'nvdiffrec_renderutils'
    # prevent ninja from using too many resources
    os.environ.setdefault('MAX_JOBS', '16')
    define_macros = []

    # Compiler options.
    opts = ['-DNVDR_TORCH']

    # Linker options.
    if os.name == 'posix':
        ldflags = ['-lcuda', '-lnvrtc']
    elif os.name == 'nt':
        ldflags = ['cuda.lib', 'advapi32.lib', 'nvrtc.lib']

    # List of sources.
    source_files = [
        'c_src/mesh.cu',
        'c_src/loss.cu',
        'c_src/bsdf.cu',
        'c_src/normal.cu',
        'c_src/cubemap.cu',
        'c_src/common.cpp',
        'c_src/torch_bindings.cpp'
    ]

    os.environ['TORCH_CUDA_ARCH_LIST'] = "5.0 6.0 6.1 7.0 7.5 8.0 8.6"

    if torch.cuda.is_available():
        print(f'Compiling {ext_name} with CUDA')
        define_macros += [('WITH_CUDA', None)]
        # op_files = glob.glob('./c_src/*')
        # extension = CUDAExtension
    else:
        raise NotImplementedError

    include_path = os.path.abspath('./c_src')
    ext_ops = CUDAExtension(
        name=ext_name,
        sources=source_files,
        include_dirs=[include_path],
        define_macros=define_macros,
        extra_compile_args=opts + ldflags,
        libraries=['cuda', 'nvrtc'],
        extra_cuda_cflags=opts,
        extra_cflags=opts,
        extra_ldflags=ldflags)
    extensions.append(ext_ops)
    return extensions

setup(
    name='nvdiffrec_renderutils',
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
    )