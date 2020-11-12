from numpy.core.multiarray import packbits
from joebvp import atomicdata
from setuptools import setup, find_packages

setup(name='joebvp',
      version='0.2',
      description=
      'Software for efficiently fitting Voigt profiles to UV spectra.',
      url='http://github.com/jnburchett/joebvp',
      author='Joseph N. Burchett',
      author_email='jburchet@astro.umass.edu',
      packages=find_packages())
# packages=['joebvp', 'joebvp.atomicdata', 'joebvp.LSF'])
