import ast
import logging

from marius.tools.mpic import astpp  # noqa: F401
from marius.tools.mpic.codegen import (
    extract_func_decls,
    extract_helper_funcs,
    extract_local_vars,
    extract_member_variables,
    extract_options,
    generate_func_body,
)
from marius.tools.mpic.symtab import SymbolTable
from marius.tools.mpic.typechecker import TypeCheckerPass
from marius.tools.mpic.utils import ClassType, camel_to_snake


def run_layer_pass(
    symbol_table: SymbolTable,
    classes: dict[ClassType],
    classdef: ast.ClassDef,
    typemap: dict[str, str],
) -> dict:
    logging.info(f"Running layer pass for {classdef.name} ...")

    # Pass1: Check semantic types
    typechecker = TypeCheckerPass(symbol_table, classes)
    typechecker.visit(classdef)
    class_attrs = classes[classdef.name]
    expr_attrs = typechecker.expr_attrs
    func_vars = typechecker.func_vars

    # TODO: Optimization: Inline and value propagate linear

    # Pass2: Extract options
    options, input_dim, output_dim = extract_options(classdef, class_attrs, typemap)

    # Pass3: Extract instance variables
    member_vars = extract_member_variables(class_attrs, typemap)

    # Pass4: Extract member function declarations
    func_decls = extract_func_decls(class_attrs, typemap)

    # Pass5: Extract local variables
    local_vars = extract_local_vars(func_vars, typemap)

    # Pass6: Generate function body
    func_body = generate_func_body(classes, classdef, expr_attrs, input_dim, output_dim)

    init_func = {"local_vars": local_vars["__init__"], "body": func_body["__init__"]}
    reset_func = {
        "local_vars": local_vars["reset_parameters"],
        "body": func_body["reset_parameters"],
    }
    forward_func = {
        "local_vars": local_vars["forward"],
        "body": func_body["forward"],
        "inputs": class_attrs["forward"].args[1].name,
        "graph": class_attrs["forward"].args[0].name,
    }
    helper_funcs = extract_helper_funcs(func_decls, local_vars, func_body)

    return {
        "header_params": {
            "LayerClassName": classdef.name,
            "options": options,
            "member_vars": member_vars,
            "member_fns": func_decls,
        },
        "source_params": {
            "layer_class_name": camel_to_snake(classdef.name),
            "LayerClassName": classdef.name,
            "init": init_func,
            "reset": reset_func,
            "forward": forward_func,
            "member_fns": helper_funcs,
        },
    }
