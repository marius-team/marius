import ast
import logging

from marius.tools.mpic import astpp  # noqa: F401
from marius.tools.mpic.attrs import GlobalAttrsPass, InstanceAttrsPass
from marius.tools.mpic.builtins import get_builtin_typemap
from marius.tools.mpic.features import FeatureFilterPass
from marius.tools.mpic.layerpass import run_layer_pass
from marius.tools.mpic.render import CodeRenderer
from marius.tools.mpic.symtab import SymbolTable
from marius.tools.mpic.utils import SemError, camel_to_snake


class LayerExtractorPass(ast.NodeVisitor):
    """
    Runs semantic checks for each GNN layer module
    Expects both a symbol table and class instance attributes
    Ensures
    - class inherits from mpi.Module and nothing else
    """

    def __init__(self, symbol_table, classes, typemap):
        self.symbol_table = symbol_table
        self.classes = classes
        self.typemap = typemap
        self.renderer = CodeRenderer()

    def visit_ClassDef(self, classdef):
        # TODO: Support inheritance
        # TODO: Support multiple classes in the same file
        if len(classdef.bases) != 1 or ast.unparse(classdef.bases[0]) != "mpi.Module":
            raise SemError(f"All classes must inherit from mpi.Module! lineno:{classdef.lineno}")

        params = run_layer_pass(self.symbol_table, self.classes, classdef, self.typemap)
        filename = camel_to_snake(classdef.name)
        self.renderer.render_header(params["header_params"], filename + ".h")
        self.renderer.render_source(params["source_params"], filename + ".cpp")

    def visit_Module(self, module):
        # XXX: cleaner to infer member variables in a separate pass
        for child in module.body:
            self.visit(child)


def populate_typemap(tree: ast.AST) -> dict[str, str]:
    typemap = get_builtin_typemap()
    for child in tree.body:
        if isinstance(child, ast.ClassDef):
            typemap[child.name] = child.name
    return typemap


def run_global_pass(tree: ast.AST):
    logging.info("Running global pass ...")

    # Pass1: Detect and disable "fancy" features
    FeatureFilterPass().visit(tree)

    # Pass2: Global attribute pass to collect all classes
    global_attrs_pass = GlobalAttrsPass()
    global_attrs_pass.visit(tree)
    global_attrs = global_attrs_pass.global_attrs
    symbol_table = SymbolTable(global_attrs)
    classes = global_attrs_pass.classes

    # Pass3: Instance attrs pass to generate object schemas
    instance_attrs_pass = InstanceAttrsPass(symbol_table, classes)
    instance_attrs_pass.visit(tree)

    # Pass4: Type check each class
    typemap = populate_typemap(tree)
    LayerExtractorPass(symbol_table, classes, typemap).visit(tree)
