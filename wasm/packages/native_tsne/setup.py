from setuptools import setup,Extension
from Cython.Build import cythonize
import numpy
extensions=[Extension('native_neighbors',['native_neighbors.pyx'],include_dirs=[numpy.get_include()],extra_compile_args=['-ffp-contract=off']),Extension('native_quad_tree',['native_quad_tree.pyx'],include_dirs=[numpy.get_include()],extra_compile_args=['-ffp-contract=off']),Extension('native_barnes_hut_tsne',['native_barnes_hut_tsne.pyx','positive_power.c'],include_dirs=[numpy.get_include()],extra_compile_args=['-ffp-contract=off'])]
setup(name='course-native-tsne',version='1.0.2',packages=[],py_modules=["neighbor_compat"],license_files=['LICENSE'],ext_modules=cythonize(extensions,compiler_directives={'cdivision':True}))
