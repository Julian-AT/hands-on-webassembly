from setuptools import setup,Extension
from Cython.Build import cythonize

setup(name='course-native-cnn',version='1.0.0',packages=[],
    license_files=['LICENSE-PYTORCH','vendor/LICENSE','vendor/LICENSE-PSIMD'],
    ext_modules=cythonize([Extension('native_cnn',
        ['native_cnn.pyx','nnpack_fft_forward.c','conv_forward.c','linear_reduction.c','native_reductions.c'],
        extra_objects=['fft.o'],extra_compile_args=['-ffp-contract=off'])],
        compiler_directives={'language_level':3}))
