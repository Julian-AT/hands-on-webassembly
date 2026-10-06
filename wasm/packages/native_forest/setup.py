from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
setup(name='course-forest-criterion',version='1.0.0',packages=[],
      ext_modules=cythonize([Extension('forest_criterion',['forest_criterion.pyx'],
          include_dirs=[numpy.get_include()],extra_compile_args=['-ffp-contract=off'])]))
