import os
from sys import platform
if platform == "linux":
    os.environ["TERRA_CONFIG_PATH"] = "/home/sabri/code/domino-21/terra_config.json"

import terra 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
from tqdm.auto import tqdm
from mosaic import DataPanel, NumpyArrayColumn