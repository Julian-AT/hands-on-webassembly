from setuptools import setup, Extension
from Cython.Build import cythonize

setup(name='course-native-neural', version='1.0.0', packages=[], license_files=['LICENSE-PYTORCH','vendor/LICENSE.txt'],
      ext_modules=cythonize([Extension('native_linear', ['native_linear.pyx','vendor/sleefsimdsp.c','vendor/rempitab.c'], include_dirs=['vendor'], define_macros=[('ENABLE_PURECFMA_SCALAR','1'),('FP_FAST_FMA','1'),('FP_FAST_FMAF','1')],
          extra_compile_args=['-ffp-contract=off'])], compiler_directives={'language_level':3}))
