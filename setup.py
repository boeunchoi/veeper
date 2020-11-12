#from numpy.core.multiarray import packbits
from joebvp import atomicdata
from setuptools import setup, find_packages
import glob
from pkg_resources import resource_filename

data_files = []
root_path = resource_filename('joebvp', 'atomicdata/')
directories = glob.glob(root_path)
for directory in directories:
    files = glob.glob(directory + '*[!_.py]')
    data_files.append((directory, files))
# then pass data_files to setup()

setup(name='joebvp',
      version='0.5',
      description=
      'Software for efficiently fitting Voigt profiles to UV spectra.',
      url='http://github.com/jnburchett/joebvp',
      author='Joseph N. Burchett',
      author_email='jburchet@astro.umass.edu',
      data_files=data_files,
      packages=find_packages())
# packages=['joebvp', 'joebvp.atomicdata', 'joebvp.LSF'])
