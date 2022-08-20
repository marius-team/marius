import os
#import torch # this import here causes a GIL error
only_python = os.environ.get("MARIUS_ONLY_PYTHON", None)
if not only_python:
    try:
        from ._pymarius import *
    except ModuleNotFoundError:
        print("Bindings not installed")
