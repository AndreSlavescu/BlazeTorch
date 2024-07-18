from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='blazetorch',
    ext_modules=[
        CppExtension(
            name='blazetorch',
            sources=['register.cc'],
            extra_compile_args=['-std=c++17'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
