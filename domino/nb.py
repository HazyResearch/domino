import os
from sys import platform

if platform == "linux":
    os.environ["TERRA_CONFIG_PATH"] = "/home/sabri/code/domino-21/terra_config.json"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import terra
import torch
import torch.nn as nn
from meerkat import DataPanel, NumpyArrayColumn
from tqdm.auto import tqdm
