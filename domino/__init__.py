from ._embed import embed, encoders
from ._slice.domino import DominoSlicer
from ._slice.spotlight import SpotlightSlicer
from ._slice.barlow import BarlowSlicer
from ._slice.multiaccuracy import MultiaccuracySlicer
from ._describe.generate import generate_candidate_descriptions
from ._describe import describe
from .gui import explore

__all__ = [
    "DominoSlicer",
    "SpotlightSlicer",
    "BarlowSlicer",
    "MultiaccuracySlicer"
    "embed",
    "encoders",
    "explore",
    "describe",
    "generate_candidate_descriptions",
]
