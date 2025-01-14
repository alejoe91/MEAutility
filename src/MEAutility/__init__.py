import importlib.metadata

__version__ = importlib.metadata.version("MEAutility")

from .core import MEA, RectMEA, Electrode, return_mea, return_mea_info, return_mea_list, add_mea, remove_mea
from .plotting import *
