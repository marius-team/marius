import os
only_python = os.environ.get("MARIUS_ONLY_PYTHON", None)
if not only_python:
    from ._pymarius import *