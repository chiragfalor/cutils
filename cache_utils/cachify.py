import functools
import os
import pickle
import pathlib
from typing import Callable, ParamSpec, TypeVar

if __package__ is None:
    from ArgsKwargs import ArgsKwargs
else:
    from .ArgsKwargs import ArgsKwargs

P = ParamSpec('P')
R = TypeVar('R')


class CachifyManager:
    """
    Manages caching for a single function. It caches the function's results to disk,
    allowing for reuse of computed results for the same arguments.
    
    Attributes:
        func (Callable): The target function to cache.
        directory (pathlib.Path): The directory where cache files are stored.
    """
    def __init__(self, func: Callable, directory: pathlib.Path):
        self.func: Callable[P, R] = func
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def cache_info(self, load_objs=False):
        """
        Returns a dictionary containing information about the cache files.
        
        Args:
            load_objs (bool, optional): If True, the cached objects will be loaded and returned.
                                        If False, only the file paths are returned. Defaults to False.
        
        Returns:
            dict: A dictionary mapping an ArgsKwargs representation to either the loaded result or the file path.
        """
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
        """
        Clears all cached files for the function.
        """
        cache_files = os.listdir(self.directory)
        for file in cache_files:
            os.remove(self.directory / file)
        print(f"Cleared cache for {self.func.__name__}")

    def cache_update(self, *args, **kwargs):
        """
        Forces an update of the cache for a specific set of arguments by re-computing the result.
        
        Args:
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        
        Returns:
            The newly computed result.
        """
        cache_file = self._get_filename(args, kwargs)
        result = self.func(*args, **kwargs)
        self._save_result(cache_file, result)
        print(f"Updated cache for {self.func.__name__} at file {cache_file}")
        return result

    def get_wrapped_func(self) -> Callable[P, R]:
        """
        Wraps the target function with caching behavior and returns the wrapped function.
        
        Returns:
            Callable[P, R]: The wrapped function that caches its results.
        
        The wrapped function will:
            - Check for a cached result on disk.
            - If available, load and return the cached result.
            - Otherwise, compute the result, cache it, and return it.
        
        The returned function also has additional attributes:
            - cache_dir: The directory where cache files are stored.
            - cache_info: Function to get cache file information.
            - cache_clear: Function to clear the cache.
            - cache_update: Function to update the cache for specific arguments.
        """
        @functools.wraps(self.func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
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
    """
    Factory function that creates a caching decorator for functions.
    
    This function returns a decorator factory that can be used to cache
    the results of function calls. It organizes cache files in a structured
    directory layout and supports versioning of cache data.
    
    Args:
        directory (str | pathlib.Path): The root directory where all cache files
                                        will be stored.
    
    Returns:
        A decorator factory function that accepts a version number.
    
    The returned decorator factory works as follows:
        - It takes an optional version parameter (default is 0) to allow for
          versioning of cached results.
        - It creates a function-specific cache directory with the structure:
            <root_directory>/<function_name>/v<version>/
        - It wraps the target function with caching behavior via CachifyManager.
    
    Example:
    >>> # Create a caching decorator with a specified root directory
    >>> cachify = cachify_factory("cache_dir/path/to/cache_directory")
    >>> # Optionally specify a version (default is 0)
    >>> @cachify(version=1)
    ... def expensive_computation(x, y):
    ...     # Simulate an expensive computation
    ...     return x * y
    >>>
    >>> # First call: computes and caches the result
    >>> result1 = expensive_computation(3, 4)
    >>> # Second call: retrieves the result from the cache
    >>> result2 = expensive_computation(3, 4)
    >>> # Inspect cache information
    >>> print(expensive_computation.cache_info())
    {ArgsKwargs(args=('3', '4'), kwargs={}): WindowsPath('cache_dir/path/to/cache_directory/expensive_computation/v1/3__4.pkl')}
    >>> # Clear the cache if needed
    >>> expensive_computation.cache_clear()
    Cleared cache for expensive_computation
    >>>
    """
    directory = pathlib.Path(directory)
    def decorator_factory(version=0):
        def cache_decorator(func: Callable[P, R]) -> Callable[P, R]:
            # Create the function-specific directory
            version_dir = directory / func.__name__ / f"v{version}"
            cm = CachifyManager(func, version_dir)
            # return the cache wrapped function
            return cm.get_wrapped_func()
        return cache_decorator
    return decorator_factory


if __name__ == "__main__":
    import doctest
    doctest.testmod()