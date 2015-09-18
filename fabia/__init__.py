from .fabia import FabiaBiclustering
from .generate_dataset import make_fabia_biclusters


import pkg_resources as __pkg_resources
__version__ = __pkg_resources.require('binet')[0].version

all = ['FabiaBiclustering', 'make_fabia_biclusters']
