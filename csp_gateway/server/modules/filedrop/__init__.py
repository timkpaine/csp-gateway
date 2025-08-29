import sys

if sys.platform.startswith("linux"):
    try:
        from .adapter import *
        from .filedrop import *
    except ImportError:
        pass
else:
    pass
