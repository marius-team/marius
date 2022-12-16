import ast

from marius.tools.mpic import astpp  # noqa: F401
from marius.tools.mpic.utils import SynError


class FeatureFilterPass(ast.NodeVisitor):
    """
    The compiler does not support the complete python grammar
    Detect and filter syntax errors before semantic analysis

    Disables
    - comprehensions
    - print
    - assert
    - lambdas
    - generators
    - coroutines
    - interactive and eval modes
    - starred variables
    - formatting strings
    """

    ##########
    # Literals
    ##########

    def visit_FormattedValue(self, node):
        raise SynError(f"Format spec is not supported! lineno:{node.lineno}")

    def visit_JoinedStr(self, node):
        raise SynError(f"Python style formatting is not supported! lineno:{node.lineno}")

    ###########
    # Variables
    ###########
    def visit_Starred(self, node):
        raise SynError(f"Starred variables are not supported! lineno:{node.lineno}")

    #############
    # Expressions
    #############
    def visit_NamedExpr(self, node):
        raise SynError(f"Named expressions are not supported! lineno:{node.lineno}")

    ##############
    # Subscripting
    ##############
    def visit_Index(self, index):
        if not isinstance(index.value, str):
            raise SynError(f"Only string indices are supported! lineno:{index.lineno}")

    def visit_Slice(self, index):
        raise SynError(f"Index slices are not supported! lineno:{index.lineno}")

    def visit_ExtSlice(self, extslice):
        raise SynError(f"Ext slices are not supported! lineno:{extslice.lineno}")

    ################
    # Comprehensions
    ################
    def visit_ListComp(self, node):
        raise SynError(f"List comprehensions are not supported! lineno:{node.lineno}")

    def visit_SetComp(self, node):
        raise SynError(f"Set comprehensions are not supported! lineno:{node.lineno}")

    def visit_GeneratorExp(self, node):
        raise SynError(f"Generator expresions are not supported! lineno:{node.lineno}")

    def visit_DictComp(self, node):
        raise SynError(f"Dict comprehensions are not supported! lineno:{node.lineno}")

    ############
    # Statements
    ############
    def AugAssign(self, node):
        raise SynError(f"Augmented Assign is not supported! lineno:{node.lineno}")

    def visit_Print(self, node):
        raise SynError(f"Print is not supported! lineno:{node.lineno}")

    def visit_Assert(self, node):
        raise SynError(f"Assert statements are not supported! lineno:{node.lineno}")

    def visit_Delete(self, node):
        raise SynError(f"Delete statements are not supported! lineno:{node.lineno}")

    #########
    # Imports
    #########
    def visit_Import(self, node):
        raise SynError(f"Imports are not supported! lineno:{node.lineno}")

    def visit_ImportFrom(self, node):
        raise SynError(f"Imports are not supported! lineno:{node.lineno}")

    ##############
    # Control Flow
    ##############
    def visit_For(self, node):
        raise SynError(f"Loops are not supported! lineno:{node.lineno}")

    def visit_While(self, node):
        raise SynError(f"Loops are not supported! lineno:{node.lineno}")

    def visit_Break(self, node):
        raise SynError(f"Loops are not supported! lineno:{node.lineno}")

    def visit_Continue(self, node):
        raise SynError(f"Loops are not supported! lineno:{node.lineno}")

    def visit_Try(self, node):
        raise SynError(f"Exception handling is not supported! lineno:{node.lineno}")

    def visit_Finally(self, node):
        raise SynError(f"Exception handling is not supported! lineno:{node.lineno}")

    def visit_Except(self, node):
        raise SynError(f"Exception handling is not supported! lineno:{node.lineno}")

    ################################
    # Function and Class Definitions
    ################################
    def visit_Lambda(self, node):
        raise SynError(f"No lambda definitions allowed! lineno:{node.lineno}")

    def visit_Yield(self, node):
        raise SynError(f"Generators not supported! lineno:{node.lineno}")

    def visit_YieldFrom(self, node):
        raise SynError(f"Generators not supported! lineno:{node.lineno}")

    def Global(self, node):
        raise SynError(f"Global not supported! lineno:{node.lineno}")

    def NonLocal(self, node):
        raise SynError(f"NonLocal not supported! lineno:{node.lineno}")

    #################
    # Async and Await
    #################
    def visit_AsyncFunctionDef(self, node):
        raise SynError(f"Coroutines not supported! lineno:{node.lineno}")

    def visit_Await(self, node):
        raise SynError(f"Coroutines not supported! lineno:{node.lineno}")

    def visit_AsyncFor(self, node):
        raise SynError(f"Coroutines not supported! lineno:{node.lineno}")

    def visit_AsyncWith(self, node):
        raise SynError(f"Coroutines not supported! lineno:{node.lineno}")
