"""
Generate cpp code from the IR
"""

import logging
import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


class CodeRenderer:
    def __init__(self):
        output_dir = Path(os.getcwd(), "build/mpic_gen").resolve()

        self.env = Environment(
            loader=FileSystemLoader(Path(Path(__file__).parent, "templates").resolve()),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
            line_comment_prefix="//*",
            line_statement_prefix="->",
        )
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            logging.info(f"Making missing directory {output_dir}")
            os.makedirs(output_dir)

    def render_header(self, params: dict, output_file: str):
        logging.info(f"Rendering header file {output_file}")
        self.render(params, "gnn_layer.jinja.h", output_file)

    def render_source(self, params: str, output_file: str):
        logging.info(f"Rendering source file {output_file}")
        self.render(params, "gnn_layer.jinja.cpp", output_file)

    def render(self, params: dict, template: str, output_file: str):
        output_file = os.path.join(self.output_dir, output_file)
        with open(output_file, "w") as outfile:
            output = self.env.get_template(template).render(**params)
            outfile.write(output)
