from ._embed import embed, encoders
from ._slice.abstract import Slicer
from ._slice.mixture import MixtureSlicer, DominoSlicer
from ._slice.spotlight import SpotlightSlicer
from ._slice.barlow import BarlowSlicer
from ._slice.multiaccuracy import MultiaccuracySlicer
from ._slice.mlp import MLPSlicer
from ._slice.fused import FusedSlicer
from ._slice.abstract import Slicer 
from ._describe.generate import generate_candidate_descriptions
from ._describe.abstract import Describer
from ._describe.mean import MeanDescriber
from ._describe.corr import CorrDescriber
from ._describe import describe
from .main import discover
from .gui import explore

__all__ = [
    "DominoSlicer",
    "MixtureSlicer",
    "MLPSlicer",
    "SpotlightSlicer",
    "BarlowSlicer",
    "MultiaccuracySlicer",
    "FusedSlicer",
    "Slicer",
    "Describer",
    "MeanDescriber",
    "CorrDescriber",
    "embed",
    "encoders",
    "explore",
    "describe",
    "discover",
    "generate_candidate_descriptions",
]
