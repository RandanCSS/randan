import sys
from contextlib import contextmanager
from functools import wraps

def _colorize(code):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sys.stdout.write(code)
            try: return func(*args, **kwargs)
            finally: sys.stdout.write("\033[0m")
        return wrapper
    return decorator

gray_prints  = _colorize("\033[90m")
green_prints = _colorize("\033[32m")
red_prints   = _colorize("\033[31m")
