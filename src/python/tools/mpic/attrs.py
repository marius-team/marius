import ast
from contextlib import contextmanager

from marius.tools.mpic.builtins import IntAttrs, NoneAttrs, get_builtin_classes
from marius.tools.mpic.symtab import SymbolTable
from marius.tools.mpic import astpp  # noqa: F401
from marius.tools.mpic.utils import Arg, Attrs, Callable, ClassType, SemError, SynError


class GlobalAttrsPass(ast.NodeVisitor):
    """
    Collect all class names in global scope
    Ensures
    - classes at the top level
    """

    def __init__(self):
        self.global_attrs = Attrs()
        self.classes = get_builtin_classes()

    def visit_ClassDef(self, classdef):
        if classdef.name in self.global_attrs:
            raise SemError(f"Class {classdef.name} is multiply defined! lineno:{classdef.lineno}")

        class_type = ClassType(classdef.name)
        self.global_attrs[classdef.name] = class_type
        # XXX: Add input_dim and output_dim as instance variables
        # TODO: add support for const int to prevent modification
        self.classes[class_type] = Attrs({"_mpic_class": class_type, "input_dim": IntAttrs, "output_dim": IntAttrs})

    def visit_Module(self, module):
        print(__file__, 32)
        for child in module.body:
            if isinstance(child, ast.ClassDef):
                self.visit(child)
            elif not isinstance(child, ast.Expr):
                # Allow comments (expressions as standalone statements)
                raise SemError(f"Only mpi.Module classes allowed at the top level! lineno:{child.lineno}")

    def generic_visit(self, node):
        raise RuntimeError(f"Internal error!\n{astpp.dump(node)}")


class InstanceAttrsPass(ast.NodeVisitor):
    """
    Generate a mapping from class names to the class attributes
    The attribute map has instance variables and instance methods
    Class variables and class methods are disallowed at the moment
    Instance variables map to its attribute map
    Instance methods map to its `Callable` object
    Expects a symbol table populated with symbols in the builtin and global scopes
    """

    def __init__(self, symbol_table: SymbolTable, classes: dict[ClassType, Attrs]):
        self.symbol_table = symbol_table
        self.classes = classes
        self.instance_attrs = None

    def visit_Name(self, name):
        attrs = self.symbol_table.find_symbol(name.id)
        if attrs is None:
            raise SemError(f"Could not find symbol {name.id}! lineno:{name.lineno}")
        return attrs

    def visit_Attribute(self, attr):
        value_attrs = self.visit(attr.value)
        if attr.attr not in value_attrs:
            raise SemError(f"Could not find symbol {attr.attr}! lineno:{attr.lineno}")
        return value_attrs[attr.attr]

    def visit_AnnAssign(self, assign):
        """
        Collect instance variables
        Ensures
        - valid type annotations
        - unique names
        """
        lno = assign.lineno

        if not isinstance(assign.target, ast.Name):
            raise SynError(f"Definition is not supported! lineno:{lno}")

        if not assign.annotation:
            raise SynError(f"Require type annotation for instance variables! lineno:{lno}")

        var_name = assign.target.id
        var_attrs = self.classes[self.visit(assign.annotation)]

        if var_name in self.instance_attrs:
            raise SemError(f"Duplicate attr definition! lineno:{assign.lineno}")

        self.instance_attrs[var_name] = var_attrs

    def visit_FunctionDef(self, func):
        """
        Collects instance functions
        Ensures
        - unique names
        - valid type annotations for all arguments and return
        - self first argument
        Disables
        - operator overriding
        - decorator lists
        - type comments
        - position only arguments
        - *args and **kwargs
        """
        lno = func.lineno

        if func.name in self.instance_attrs:
            raise SemError(f"Duplicate attr definition! func: lineno:{lno}")

        if (
            func.name != "__init__"
            and func.name != "__call__"
            and func.name.startswith("__")
            and func.name.endswith("__")
        ):
            raise SynError(f"Operator overriding is not supported! lineno:{lno}")

        if func.name == "reset":
            raise SynError(f"Please use reset_parameters to reset torch params! lineno:{lno}")

        if func.decorator_list:
            raise SynError(f"Decorator lists are not supported! lineno:{lno}")

        if func.type_comment:
            raise SynError(f"Type comments are not supported! lineno:{lno}")

        if func.args.vararg:
            raise SynError(f"*args is not supported! lineno:{lno}")

        if func.args.kwarg:
            raise SynError(f"**kwargs is not supported! func: lineno:{lno}")

        if func.args.posonlyargs or func.args.kwonlyargs:
            raise SynError(f"pos only or kw only args are not supported! lineno:{lno}")

        if func.args.kw_defaults or func.args.defaults:
            raise SynError(f"Defaults for arguments are not supported! lineno:{lno}")

        if func.args.args[0].arg != "self":
            raise SynError(f"Only instance methods are supported! lineno:{lno}")

        args = []
        for arg in func.args.args[1:]:
            if not arg.annotation:
                raise SynError(f"Type annotations are required! lineno:{lno}")
            arg_attrs = self.classes[self.visit(arg.annotation)]
            args.append(Arg(arg.arg, arg_attrs))

        if func.returns:
            return_attrs = self.classes[self.visit(func.returns)]
        else:
            return_attrs = NoneAttrs
        self.instance_attrs[func.name] = Callable(args, return_attrs)

    def visit_ClassDef(self, classdef):
        """
        Populates class attributes
        """
        if classdef.decorator_list:
            raise SemError(f"Decorator lists are not supported! lineno:{classdef.lineno}")

        if classdef.keywords:
            raise SemError(f"Class arguments are not supported! lineno:{classdef.lineno}")

        @contextmanager
        def managed_instance_attrs():
            self.instance_attrs = self.classes[ClassType(classdef.name)]
            yield
            self.instance_attrs = None

        with managed_instance_attrs():
            for child in classdef.body:
                self.visit(child)

    def visit_Module(self, module):
        for child in module.body:
            if isinstance(child, ast.ClassDef):
                self.visit(child)
            elif not isinstance(child, ast.Expr):
                # Allow comments
                raise SemError(f"Only mpi.Module classes allowed at the top level! lineno:{child.lineno}")

    def generic_visit(self, node):
        raise RuntimeError(f"Internal error!\n{astpp.dumps(node)}")
