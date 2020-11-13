#from numpy.core.multiarray import packbits
# from joebvp import atomicdata
from setuptools import setup, find_packages
import glob
from pkg_resources import resource_filename
import os
import pathlib


def package_files(directory):
    directory = str(pathlib.Path(__file__).parent.absolute()) + str(
        pathlib.Path(directory))
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            if '__' not in filename:
                paths.append(os.path.join('..', path, filename))
    return paths


data_files = package_files('/joebvp/atomicdata/')

setup(name='joebvp',
      version='0.5',
      description=
      'Software for efficiently fitting Voigt profiles to UV spectra.',
      url='http://github.com/jnburchett/joebvp',
      author='Joseph N. Burchett',
      author_email='jburchet@astro.umass.edu',
      package_data={'joebvp': data_files},
      packages=find_packages())
