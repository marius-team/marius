import ast
from contextlib import contextmanager
from functools import wraps

from marius.tools.mpic import astpp  # noqa: F401
from marius.tools.mpic.builtins import (
    DENSEGraphAttrs,
    EdgeDataAttrs,
    GraphLocalScopeAttrs,
    IntAttrs,
    LayerAttrs,
    LinearAttrs,
    NodeDataAttrs,
    NoneAttrs,
    StrAttrs,
    TensorAttrs,
    edge_data_prefix,
    is_consistent_with,
    is_numeric,
    node_data_prefix,
)
from marius.tools.mpic.symtab import SymbolTable
from marius.tools.mpic.utils import Attrs, Callable, ClassType, SemError, SynError


def update_expr_attrs(visit_expr):
    @wraps(visit_expr)
    def wrapper(typechecker, node):
        expr_attrs = visit_expr(typechecker, node)
        typechecker.expr_attrs[node] = expr_attrs
        return expr_attrs

    return wrapper


class TypeCheckerPass(ast.NodeVisitor):
    """
    visit_className returns object attributes of the expression, None otherwise
    Runs semantic checks
    Registers new instance variables introduced in __init__ using type inference
    Ensures
    - functions are called with valid arguments
    - called symbols are callable
    - new instance variables introduced in __init__ are mapped
    - assignments are on consistent types
    - expressions are properly typed
    - operators are called with compatible types
    Disables
    - named expressions
    - nested functions or classes
    - registering new variables with non-enclosing scopes
    - registering new variables in non __init__ functions
    - generic subscripting
    Trampolines
    - constructors using __init__
    - callable objects using __call__
    """

    def __init__(self, symbol_table: SymbolTable, classes: dict[str, Attrs]):
        self.symbol_table = symbol_table
        self.classes = classes
        self.expr_attrs = dict()
        self.self_attrs = None
        self.init_mode = False
        self.return_attrs = None
        self.func_vars = dict()
        self.current_locals = None
        self.graph_local_scope = 0

    ##########
    # Literals
    ##########
    @update_expr_attrs
    def visit_Constant(self, constant):
        if isinstance(constant.value, int):
            return IntAttrs
        elif isinstance(constant.value, str):
            return StrAttrs
        elif isinstance(constant.value, Ellipsis):
            return None
        else:
            raise SemError(f"Literal is not supported! lineno:{constant.lineno}")

    @update_expr_attrs
    def visit_List(self, node):
        raise SynError(f"Lists are not supported! lineno:{node.lineno}")

    @update_expr_attrs
    def visit_Tuple(self, node):
        raise SynError(f"Tuples are not supported! lineno:{node.lineno}")

    @update_expr_attrs
    def visit_Set(self, node):
        raise SynError(f"Sets are not supported! lineno:{node.lineno}")

    @update_expr_attrs
    def visit_Dict(self, node):
        raise SynError(f"Dicts are not supported! lineno:{node.lineno}")

    ###########
    # Variables
    ###########
    @update_expr_attrs
    def visit_Name(self, name):
        return self.symbol_table.find_symbol(name.id)

    #############
    # Expressions
    #############
    def visit_Expr(self, expr):
        self.visit(expr.value)

    @update_expr_attrs
    def visit_UnaryOp(self, node):
        raise SynError(f"Unary operations are not supported! lineno:{node.lineno}")

    @update_expr_attrs
    def visit_BinOp(self, bin_op):
        if not isinstance(bin_op.op, ast.Mult):
            raise SynError(f"Unexpected operation! lineno:{bin_op.lineno}")

        if not is_numeric(self.visit(bin_op.left)):
            raise SemError(f"Unexpected operand! lineno:{bin_op.lineno}")

        if not is_numeric(self.visit(bin_op.right)):
            raise SemError(f"Unexpected operand! lineno:{bin_op.lineno}")

        return IntAttrs

    @update_expr_attrs
    def visit_BoolOp(self, node):
        raise SynError(f"bool ops are not supported! lineno:{node.lineno}")

    @update_expr_attrs
    def visit_Compare(self, node):
        raise SynError(f"Compare ops not supported! lineno:{node.lineno}")

    @update_expr_attrs
    def visit_Call(self, call):
        call_attrs = self.visit(call.func)
        fqname = ast.unparse(call.func)
        lno = call.lineno
        return_attrs = None

        # Extract the type of callable
        if fqname == "super":
            # FIXME: semantic checking for call to super
            return LayerAttrs
        elif isinstance(call_attrs, ClassType):
            # Constructor
            class_type = call_attrs
            class_attrs = self.classes[class_type]
            func_sig = class_attrs["__init__"]
            return_attrs = class_attrs
        elif isinstance(call_attrs, Attrs):
            # callable object
            if "__call__" not in call_attrs:
                raise SemError(f"Expression {fqname} is not callable! lineno:{lno}")
            func_sig = call_attrs["__call__"]
        else:
            func_sig = call_attrs

        if not isinstance(func_sig, Callable):
            raise SemError(f"Expression {fqname} is not callable! lineno:{lno}")

        # Verify function call arguments
        formal_args = [arg for arg in reversed(func_sig.args)]  # reversed to pop args from behind
        if len(formal_args) < len(call.args):
            raise SemError(f"Function called with invalid number of arguments! lineno:{lno}")
        for actual_arg in call.args:
            # Verify positional arguments
            actual_attrs = self.visit(actual_arg)
            if not is_consistent_with(actual_attrs, formal_args[-1].attrs):
                aname = ast.unparse(actual_arg)
                raise SemError(f"Function called with invalid arg {aname}! lineno:{lno}")

            formal_args.pop()

        # Expect the rest of the arguments to be called using kw args
        if len(formal_args) != len(call.keywords):
            raise SemError(f"Function called with invalid number of arguments! lineno:{lno}")
        formal_args = {arg.name: arg.attrs for arg in formal_args}
        for keyword in call.keywords:
            aname = keyword.arg
            actual_attrs = self.visit(keyword.value)
            if aname not in formal_args:
                raise SemError(f"keyword not available in remaining arguments! lineno:{lno}")

            if not is_consistent_with(actual_attrs, formal_args[aname]):
                raise SemError(f"Function called with invalid arg {aname}! lineno:{lno}")

            del formal_args[aname]

        if call_attrs == DENSEGraphAttrs["update_all"]:
            # XXX: assume copy_u and mean
            message_func = None
            if len(call.args) > 0:
                message_func = call.args[0]
            reduce_func = None
            if len(call.args) > 1:
                reduce_func = call.args[1]
            for keyword in call.keywords:
                if keyword.arg == "message_func":
                    message_func = keyword.value
                elif keyword.arg == "reduce_func":
                    reduce_func = keyword.value
            copysrc, copydst = message_func.args[0], message_func.args[1]
            meansrc, meandst = reduce_func.args[0], reduce_func.args[1]
            if (
                not isinstance(copysrc, ast.Constant)
                or not isinstance(copysrc.value, str)
                or not isinstance(copydst, ast.Constant)
                or not isinstance(copydst.value, str)
                or not isinstance(meansrc, ast.Constant)
                or not isinstance(meansrc.value, str)
                or not isinstance(meandst, ast.Constant)
                or not isinstance(meandst.value, str)
                or copydst.value != meansrc.value
            ):
                raise SemError(f"Invalid update_all invocation! lineno:{lno}")
            # Add meandst to the list of local variables
            var_name = node_data_prefix + meandst.value
            self.symbol_table.add_symbol(var_name, NodeDataAttrs)
            self.current_locals[var_name] = NodeDataAttrs

        if call_attrs is LayerAttrs["__init__"]:
            if not isinstance(call.args[0], ast.Name) or not isinstance(call.args[1], ast.Name):
                raise SynError(f"Must pass a name! lineno{lno}")

        # Type of the call expression is the return type of the call expression
        # XXX: return_attrs handles the special case of a constructor
        return return_attrs if return_attrs else func_sig.return_attrs

    @update_expr_attrs
    def visit_IfExp(self, node):
        raise SynError(f"if-else is not supported! lineno:{node.lineno}")

    @update_expr_attrs
    def visit_Attribute(self, attr):
        value_attrs = self.visit(attr.value)
        if attr.attr in value_attrs:
            return value_attrs[attr.attr]

        return None

    ##############
    # Subscripting
    ##############
    def get_graph_local_name(self, subscript):
        if not self.graph_local_scope:
            raise SemError(f"Not in graph local scope! lineno:{subscript.lineno}")
        graph_attrs = self.visit(subscript.value)
        if graph_attrs is NodeDataAttrs:
            return node_data_prefix + subscript.slice.value
        elif graph_attrs is EdgeDataAttrs:
            return edge_data_prefix + subscript.slice.value
        else:
            raise SemError(f"Generic subscripts are not supported! lineno:{subscript.lineno}")

    @update_expr_attrs
    def visit_Subscript(self, subscript):
        return self.symbol_table.find_symbol(self.get_graph_local_name(subscript))

    ############
    # Statements
    ############
    def visit_Assign(self, assign):
        """
        Implements core logic of automatically inferring new local and instance variables
        """
        lno = assign.lineno
        if len(assign.targets) != 1:
            raise SynError(f"Multi-assignments are not supported! {lno}")

        target = assign.targets[0]
        value_attrs = self.visit(assign.value)

        formal_attrs = self.visit(target)
        if formal_attrs is not None:
            # Definition already exists!
            if not is_consistent_with(value_attrs, formal_attrs):
                raise SemError(f"Type mismatch: cannot assign to variable! lineno:{lno}")
        elif isinstance(target, ast.Name):
            # register a new local variable
            if target.id.endswith("_"):
                raise SynError(f"Variables cannot have a trailing underscore! lineno:{lno}")
            self.symbol_table.add_symbol(target.id, value_attrs)
            self.current_locals[target.id] = value_attrs
        elif isinstance(target, ast.Attribute):
            # Only allow self.<attr> = <value> to register new varaibles
            if ast.unparse(target.value) != "self":
                raise SemError(f"Cannot add symbols to non-owning scopes! lineno:{lno}")
            if not self.init_mode:
                raise SemError(f"Cannot register new instance variables outside of __init__! lineno:{lno}")
            if target.attr in {"input_dim", "output_dim"}:
                raise SynError(f"Cannot modify input_dim and output_dim! lineno:{lno}")
            if target.attr.endswith("_"):
                raise SynError(f"Variables cannot have a trailing underscore! lineno:{lno}")
            self.self_attrs[target.attr] = value_attrs
            if value_attrs is LinearAttrs:
                self.self_attrs[f"_mpic_{target.attr}_input_dim"] = IntAttrs
                self.self_attrs[f"_mpic_{target.attr}_output_dim"] = IntAttrs
        elif isinstance(target, ast.Subscript):
            # graph.ndata['h'] = ...
            local_name = self.get_graph_local_name(target)
            if local_name.endswith("_"):
                raise SynError(f"Variables cannot have a trailing underscore! lineno:{lno}")
            self.symbol_table.add_symbol(local_name, value_attrs)
            self.current_locals[local_name] = value_attrs

    def visit_AnnAssign(self, assign):
        if self.return_attrs:
            raise SemError(f"Annotated assignments are not supported in func body! lineno:{assign.lineno}")

    def visit_Raise(self, node):
        raise SynError(f"Cannot raise exceptions! lineno:{node.lineno}")

    def visit_Pass(self, _):
        pass

    ##############
    # Control Flow
    ##############
    def visit_If(self, node):
        raise SemError(f"Control flow is not supported! lineno:{node.lineno}")

    ################################
    # Function and Class Definitions
    ################################
    def visit_FunctionDef(self, func):
        if self.return_attrs:
            raise SemError(f"Nested functions are not supported! lineno:{func.lineno}")

        func_sig = self.self_attrs[func.name]

        @contextmanager
        def managed_returns():
            self.return_attrs = func_sig.return_attrs
            yield
            self.return_attrs = None

        @contextmanager
        def managed_args():
            # add self
            self.symbol_table.add_symbol("self", self.self_attrs)
            # add other args
            for arg in func_sig.args:
                self.symbol_table.add_symbol(arg.name, arg.attrs)
            yield
            # XXX: symbols removed automatically through managed scope
            ...

        @contextmanager
        def managed_locals():
            self.current_locals = dict()
            yield
            self.func_vars[func.name] = self.current_locals
            self.current_locals = None

        with managed_returns(), self.symbol_table.managed_scope(), managed_args(), managed_locals():
            for child in func.body:
                self.visit(child)

    def visit_With(self, with_node):
        lno = with_node.lineno
        if len(with_node.items) != 1:
            raise SemError(f"General case `with` is not supported! lineno:{lno}")

        with_item = with_node.items[0]
        if with_item.optional_vars:
            raise SemError(f"General case `with` is not supported! lineno:{lno}")

        if self.visit(with_item.context_expr) != GraphLocalScopeAttrs:
            raise SemError(f"General case `with` is not supported! lineno:{lno}")

        @contextmanager
        def managed_graph_local_scope():
            self.graph_local_scope += 1
            yield
            self.graph_local_scope -= 1

        with managed_graph_local_scope():
            for stmt in with_node.body:
                self.visit(stmt)

    def visit_Return(self, return_node):
        return_attrs = self.visit(return_node.value) if return_node.value else NoneAttrs
        if not is_consistent_with(return_attrs, self.return_attrs):
            raise SemError(f"Return type does not match! lineno:{return_node.lineno}")

    def visit_ClassDef(self, classdef):
        lno = classdef.lineno

        if self.self_attrs or self.return_attrs:
            raise SemError(f"Nested classes are not supported! lineno:{lno}")

        self_attrs = self.classes[ClassType(classdef.name)]

        # verify the presence of mandatory functions
        if "__init__" not in self_attrs or "reset_parameters" not in self_attrs or "forward" not in self_attrs:
            raise SemError(f"Must define __init__, reset_parameters, and forward! lineno:{lno}")
        reset_func = self_attrs["reset_parameters"]
        forward_func = self_attrs["forward"]

        # verify the signature of the reset_parameters function
        if len(reset_func.args) != 0 or not is_consistent_with(reset_func.return_attrs, NoneAttrs):
            raise SemError(f"Invalid signature of reset_parameters! lineno:{lno}")

        # verify the signature of the forward function
        if (
            len(forward_func.args) != 2
            or not is_consistent_with(forward_func.args[0].attrs, DENSEGraphAttrs)
            or not is_consistent_with(forward_func.args[1].attrs, TensorAttrs)
            or not is_consistent_with(forward_func.return_attrs, TensorAttrs)
        ):
            raise SemError(f"Invalid signature of forward! lineno:{lno}")

        @contextmanager
        def managed_self_attrs():
            self.self_attrs = self_attrs
            yield
            self.self_attrs = None

        @contextmanager
        def managed_init():
            self.init_mode = True
            yield
            self.init_mode = False

        with managed_self_attrs():
            # Pass1: process __init__
            with managed_init():
                for child in classdef.body:
                    if child.name == "__init__":
                        self.visit(child)

            # Pass2: process other nodes
            for child in classdef.body:
                if child.name != "__init__":
                    self.visit(child)

    def generic_visit(self, node):
        raise RuntimeError(f"Internal error!\n{astpp.dump(node)}")
