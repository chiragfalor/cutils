import functools
import os
import pickle
import pathlib
from typing import Callable

from .ArgsKwargs import ArgsKwargs


class CachifyManager:
    def __init__(self, func: Callable, directory: pathlib.Path):
        self.func = func
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def cache_info(self, load_objs=False):
        cache_files = os.listdir(self.directory)
        output = {}
        for file in cache_files:
            # get filename without extension
            filename = file.split(".")[0]
            akw = ArgsKwargs.from_repn(filename)
            file_path = self.directory / file
            output[akw] = self._load_result(file_path) if load_objs else file_path
        return output
    
    def cache_clear(self):
        cache_files = os.listdir(self.directory)
        for file in cache_files:
            os.remove(self.directory / file)
        print(f"Cleared cache for {self.func.__name__}")

    def cache_update(self, *args, **kwargs):
        cache_file = self._get_filename(args, kwargs)
        result = self.func(*args, **kwargs)
        self._save_result(cache_file, result)
        print(f"Updated cache for {self.func.__name__} at file {cache_file}")
        return result

    def get_wrapped_func(self):
        @functools.wraps(self.func)
        def wrapper(*args, **kwargs):
            cache_file = self._get_filename(args, kwargs)

            try: # Check if the result is already cached
                result = self._load_result(cache_file)
            except FileNotFoundError:
                result = self.func(*args, **kwargs)
                self._save_result(cache_file, result)
                
            return result
        
        wrapper.cache_dir = self.directory
        wrapper.cache_info = self.cache_info
        wrapper.cache_clear = self.cache_clear
        wrapper.cache_update = self.cache_update

        return wrapper

    def _get_filename(self, args: tuple, kwargs: dict):
        akw = ArgsKwargs(args, kwargs)
        filename = akw.get_repn() + ".pkl"
        return self.directory / filename
    
    def _save_result(self, cache_file, result):
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

    def _load_result(self, cache_file):
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Cache file {cache_file} not found.")



def cachify_factory(directory: str | pathlib.Path):
    directory = pathlib.Path(directory)
    def decorator_factory(version=0):
        def cache_decorator(func):
            # Create the function-specific directory
            version_dir = directory / func.__name__ / f"v{version}"
            cm = CachifyManager(func, version_dir)
            # return the cache wrapped function
            return cm.get_wrapped_func()
        return cache_decorator
    return decorator_factory