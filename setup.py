#from distutils.core import setup
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize

from codecs import open
from os import path
import numpy as np
VERSION = '2015.9'


# Get the long description from the relevant file
with open(path.join('README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()


setup(
    name='fabia',

    version=VERSION,

    description='Fabia Biclustering Algorithm',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/untom/fabia',

    # Author details
    author='Thomas Unterthiner',
    author_email='thomas.unterthiner@gmx.net',

    # Choose your license
    license='GPL v2',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=['Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],

    keywords='Machine Learning Biclustering Fabia',

    packages=['fabia'],
    ext_modules = cythonize([Extension("fabia._fabia", ["fabia/_fabia.pyx"],
                                       include_dirs=[np.get_include()],
              )
    ]),
    #install_requires=["scipy>=0.16.0", "scikit-learn>=0.16.1"],
)
