#!/usr/bin/env python

import argparse
import logging

from marius.tools.mpic.compiler import run_compiler


def parse_args():
    """
    positional args
    - pyfiles: list of files to compile
    optional args
    - -l/--log: log level (see https://docs.python.org/3/library/logging.html#logging-levels)
    """
    parser = argparse.ArgumentParser(prog="mpic", description="MariusGNN MPI compiler")
    parser.add_argument("pyfiles", nargs="+")
    parser.add_argument("-l", "--log", default="WARNING")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    log_level = getattr(logging, args.log.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {args.log}")

    logging.basicConfig(level=log_level, format="[%(levelname)s]\t%(message)s")

    # Compile each file
    for filename in args.pyfiles:
        run_compiler(filename)


if __name__ == "__main__":
    main()
