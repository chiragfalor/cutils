

class PipedFn:
    def __init__(self, fn, args, kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

        self.__name__ = fn.__name__

    def __rrshift__(self, other):
        return self.fn(other, *self.args, **self.kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
    
    def __repr__(self):
        return f"<PipedFn '{self.__name__}(args={self.args}, kwargs={self.kwargs})'>"



def piped(fn):
    """Decorator to pipe the result of a function into another function.

    Example:
    >>> @piped
    ... def add(x, y):
    ...     return x + y
    >>> @piped
    ... def square(x):
    ...     return x**2
    >>> 2 >> add(3) >> square()
    25
    """
    def wrapper(*args, **kwargs):
        return PipedFn(fn, args, kwargs)
    return wrapper


if __name__ == "__main__":
    import doctest
    doctest.testmod()