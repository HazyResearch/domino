from dataclasses import dataclass
from functools import reduce, wraps
from inspect import getcallargs
from typing import Collection, Mapping
import pandas as pd
import torch
import numpy as np
from typing import List

import meerkat as mk


def unpack_args(data: mk.DataPanel, *args):
    if any(map(lambda x: isinstance(x, str), args)) and data is None:
        raise ValueError("If args are strings, `data` must be provided.")

    new_args = []
    for arg in args:
        if isinstance(arg, str):
            arg = data[arg]
        if isinstance(arg, mk.AbstractColumn):
            # this is necessary because torch.tensor() of a NumpyArrayColumn is very
            # slow and I don't want implementers to have to deal with casing on this
            arg = arg.data
        new_args.append(arg)
    return new_args


def convert_to_numpy(*args):
    """Convert Torch tensors and Pandas Series to numpy arrays."""
    new_args = []
    for arg in args:
        if torch.is_tensor(arg):
            new_args.append(arg.numpy())
        elif isinstance(arg, pd.Series):
            new_args.append(arg.values)
        elif isinstance(arg, List):
            new_args.append(np.array(arg))
        else:
            new_args.append(arg)

    return tuple(new_args)

def convert_to_torch(*args):
    new_args = []
    for arg in args:
        if isinstance(arg, (np.ndarray, pd.Series, List)):
            new_args.append(torch.tensor(arg))
        else:
            new_args.append(arg)
        
    return tuple(new_args)

def nested_getattr(obj, attr, *args):
    """Get a nested property from an object.

    # noqa: E501
    Source: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    """
    return reduce(lambda o, a: getattr(o, a, *args), [obj] + attr.split("."))


@dataclass
class VariableColumn:
    variable_name: str

    def resolve(self, args_dict: dict):
        path = self.variable_name.split(".")
        obj = args_dict[path[0]]
        if len(path) > 1:
            return nested_getattr(obj, ".".join(path[1:]))
        return obj


def requires_columns(dp_arg: str, columns: Collection[str]):
    def _requires(fn: callable):
        @wraps(fn)
        def _wrapper(*args, aliases: Mapping[str, str] = None, **kwargs):
            args_dict = getcallargs(fn, *args, **kwargs)
            if "kwargs" in args_dict:
                args_dict.update(args_dict.pop("kwargs"))
            dp = args_dict[dp_arg]
            if aliases is not None:
                dp = dp.view()
                for column, alias in aliases.items():
                    dp[column] = dp[alias]

            # resolve variable columns
            resolved_cols = [
                (col.resolve(args_dict) if isinstance(col, VariableColumn) else col)
                for col in columns
            ]

            missing_cols = [col for col in resolved_cols if col not in dp]
            if len(missing_cols) > 0:
                raise ValueError(
                    f"DataPanel passed to `{fn.__qualname__}` at argument `{dp_arg}` "
                    f"is missing required columns `{missing_cols}`."
                )
            args_dict[dp_arg] = dp
            return fn(**args_dict)

        return _wrapper

    return _requires
