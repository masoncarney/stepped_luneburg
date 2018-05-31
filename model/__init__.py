"""
This is a ray tracing module that creates a model of a stepped Luneburg lens and propagates rays through the lens to produce an image.
"""

__author__ = "Mason Carney"
__version__ = "1.0"
__email__ = "masoncarney@strw.leidenuniv.nl"
__status__ = "Development"


from luneburg_lens import stepped_luneburg
from enclosed_intensity import enc_int
from intensity_map import int_map