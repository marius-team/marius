"""
Compiles the input marius script file into a header and a source file

Tools
- ast:      used for parsing marius script
- Jinja2:   used as for the code skeleton
"""

import ast
import logging

from marius.tools.mpic.globalpass import run_global_pass
from marius.tools.mpic.utils import SynError


def run_compiler(filename: str):
    with open(filename, "r") as modf:
        logging.info(f"Generating AST from file {filename} ...")
        contents = modf.read()
        if "_mpic_" in contents:
            # Reserve _mpic_ to not appear anywhere in the program
            raise SynError("Pattern _mpic_ is reserved and cannot appear anywhere in the program!")
        tree = ast.parse(contents)
        run_global_pass(tree)
