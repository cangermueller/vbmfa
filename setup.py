#!/usr/bin/env python

from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='vbmfa',
      version='0.0.1',
      description='Variational Bayesian Mixture of Factor Analysers',
      long_description=read('README.rst'),
      author='Christof Angermueller',
      author_email='cangermueller@gmail.com',
      license='GNU GPLv3+',
      url='https://github.com/cangermueller/vbmfa',
      packages=['vbmfa'],
      install_requires=['numpy>=1.8.2',
                        'scipy>=0.14.0',
                        'scikit-learn>=0.15.1'],
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Education',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                   'Operating System :: MacOS',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.4',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Bio-Informatics',
                   'Topic :: Scientific/Engineering :: Image Recognition',
                   'Topic :: Scientific/Engineering :: Information Analysis',
                   ],
      keywords=['Factor Analysis', 'PCA', 'Probabilistic PCA',
                'Dimensionality Reduction', 'Clustering'],
      )

