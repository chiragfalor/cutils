
from dataclasses import dataclass
from typing import Any
import pandas as pd
import numpy as np
import xarray as xr
import hashlib

def hash_parameter(param: Any):
    if isinstance(param, pd.DataFrame):
        return f"DataFrame_{hashlib.md5(param.to_records().tobytes()).hexdigest()}"
    elif isinstance(param, np.ndarray):
        return f"ndarray_{hashlib.md5(param.tobytes()).hexdigest()}"
    elif isinstance(param, pd.Series):
        return f"Series_{hashlib.md5(param.to_xarray().to_netcdf()).hexdigest()}"
    elif isinstance(param, xr.DataArray):
        return f"DataArray_{hashlib.md5(param.to_netcdf()).hexdigest()}"
    else:
        return repr(param)

@dataclass
class ArgsKwargs:
    args: tuple
    kwargs: dict

    def __repr__(self):
        return f"ArgsKwargs(args={self.args}, kwargs={self.kwargs})"
    
    def get_repn(self, hash_fn=hash_parameter):
        """Format the function arguments into a string suitable for filenames."""
        parts = [hash_fn(arg) for arg in self.args] + [f"{k!r}={hash_fn(v)}" for k, v in self.kwargs.items()]

        if len(parts) == 0:
            return "empty_args_kwargs"

        return "__".join(parts).replace("/", "_").replace("\\", "_")
    
    def __hash__(self):
        return hash(self.get_repn())
    
    @classmethod
    def from_repn(cls, repn: str):
        if (repn == "empty_args_kwargs") or (repn is None) or (len(repn) == 0):
            return cls((), {})

        parts = repn.split("__")
        args = []
        kwargs = {}
        for part in parts:
            if "=" in part:
                key, val = part.split("=")
                kwargs[key] = val
            else:
                args.append(part)
        return cls(tuple(args), kwargs)