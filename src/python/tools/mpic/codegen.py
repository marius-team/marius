import ast

from marius.tools.mpic import astpp  # noqa: F401
from marius.tools.mpic.builtins import DENSEGraphAttrs, LinearAttrs, builtin_attrs, node_data_prefix
from marius.tools.mpic.utils import Attrs, Callable, ClassType

instance_var_suffix = "_"


def extract_options(classdef: ast.ClassDef, class_attrs: Attrs, typemap: dict[str, str]) -> (dict[str, str], str, str):
    """
    Add all options except input_dim and output_dim
    input_dim and output_dim are inferred from call to super
    """
    # extract __init__
    init_func = None
    for child in classdef.body:
        if isinstance(child, ast.FunctionDef) and child.name == "__init__":
            init_func = child

    input_dim = None
    output_dim = None
    for child in init_func.body:
        if ast.unparse(child).startswith("super"):
            args = child.value.args
            input_dim = args[0].id
            output_dim = args[1].id

    options = dict()
    exclude = {input_dim, output_dim}
    init_sig = class_attrs["__init__"]
    for arg in init_sig.args:
        if arg.name not in exclude:
            options[arg.name] = typemap[arg.attrs["_mpic_class"]]

    return options, input_dim, output_dim


def extract_member_variables(class_attrs: Attrs, typemap: dict[str, str]) -> dict[str, str]:
    member_vars = dict()
    for name, attrs in class_attrs.items():
        if name != "_mpic_class" and name not in {"input_dim", "output_dim"} and isinstance(attrs, Attrs):
            member_vars[name + instance_var_suffix] = typemap[attrs["_mpic_class"]]
    return member_vars


def extract_func_decls(class_attrs: Attrs, typemap: dict[str, str]) -> list[dict]:
    func_decls = []
    for name, attrs in class_attrs.items():
        if (
            "__mpic_" not in name
            and name not in {"__init__", "reset_parameters", "forward"}
            and isinstance(attrs, Callable)
        ):
            returns = typemap[attrs.return_attrs["_mpic_class"]]
            arg_types = [typemap(arg.attrs["_mpic_class"]) for arg in attrs.args]
            arg_names = [arg.name for arg in attrs.args]
            args = [f"{arg_type} {name}" for arg_type, name in zip(arg_types, arg_names)]
            func_decls.append({"returns": returns, "name": name, "args": args})
    return func_decls


def extract_local_vars(func_vars: dict[str, dict[str, Attrs]], typemap: dict[str, str]) -> dict[str, dict[str, str]]:
    return {
        func_name: {name: typemap[attrs["_mpic_class"]] for name, attrs in local_vars.items()}
        for func_name, local_vars in func_vars.items()
    }


class FunctionGenerator(ast.NodeVisitor):
    """
    visit_astNode
    - returns the equaivalent cpp source for expressions
    - writes out statements with indentation
    local_vars stores all the local variables
    TODO: Add support for splitting long lines mirroring python code
    """

    def __init__(self, classes, class_attrs, expr_attrs, input_dim, output_dim):
        self.classes = classes
        self.class_attrs = class_attrs
        self.expr_attrs = expr_attrs
        self.statements = []
        self.indentation = 0
        self.input_dim = input_dim
        self.output_dim = output_dim

    def write(self, stmt: str):
        self.statements.append("    " * self.indentation + stmt)

    ##########
    # Literals
    ##########
    def visit_Constant(self, constant):
        return str(constant.value)

    ###########
    # Variables
    ###########
    def visit_Name(self, name):
        if name.id == self.input_dim:
            return "input_dim_"
        if name.id == self.output_dim:
            return "output_dim_"
        if name.id == "self":
            return "(*this)"
        return name.id

    #############
    # Expressions
    #############
    def visit_Expr(self, expr):
        if not isinstance(expr.value, ast.Constant):
            # Ignore function comments
            self.write(self.visit(expr.value) + ";")

    def visit_BinOp(self, bin_op):
        # TODO: assign priorities to operators and bracket them accordingly
        # TODO: Support more operators
        left_expr = self.visit(bin_op.left)
        right_expr = self.visit(bin_op.right)
        return f"{left_expr} * {right_expr}"

    def visit_Call(self, call):
        func_expr = self.visit(call.func)

        func_attrs = self.expr_attrs[call.func]
        if isinstance(func_attrs, ClassType):
            func_attrs = self.classes[func_attrs]["__init__"]
        elif isinstance(func_attrs, Attrs):
            func_attrs = func_attrs["__call__"]
        pos_args = [self.visit(arg) for arg in call.args]
        keywords = {keyword.arg: self.visit(keyword.value) for keyword in call.keywords}
        keyword_args = [keywords[arg.name] for arg in func_attrs.args[len(pos_args) :]]  # noqa: E203
        args = pos_args + keyword_args

        if func_attrs is DENSEGraphAttrs["update_all"]:
            # XXX: must be DENSEGraph::update_all
            # TODO: support user defined functions
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
            copysrc = message_func.args[0].value
            meandst = reduce_func.args[1].value
            inputs = node_data_prefix + copysrc
            outputs = node_data_prefix + meandst
            graph = self.visit(call.func.value)
            return f"{outputs} = update_all<MeanFunc>({graph}, {inputs});"
        elif func_attrs is builtin_attrs["mpi"]["cat"]:
            return f"torch::cat({{{args[0]}, {args[1]}}}, {args[2]})"
        elif func_attrs is LinearAttrs["reset_parameters"]:
            target = self.visit(call.func.value)
            linear_input_dim = f"_mpic_{target}input_dim_"
            linear_output_dim = f"_mpic_{target}output_dim_"
            dims = f"{{{linear_output_dim}, {linear_input_dim}}}"
            init_tensor = f"initialize_tensor(config_->init, {dims}, tensor_options)"
            autograd = f"{init_tensor}.set_requires_grad(true)"
            register = f'register_parameter("{target}", {autograd})'
            return f"{target} = {register};"
        elif func_attrs is LinearAttrs["__call__"]:
            target = self.visit(call.func)
            inputs = self.visit(call.args[0])
            return f"torch::matmul({target}, {inputs}.transpose(0, -1)).transpose(0, -1)"
        else:
            args_expr = ", ".join(args)
            return f"{func_expr}({args_expr})"

    def visit_Attribute(self, attr):
        if ast.unparse(attr.value) == "self":
            if isinstance(self.class_attrs[attr.attr], Attrs):
                # Instance variables are suffixed using an underscore
                return attr.attr + instance_var_suffix
            else:
                return attr.attr if attr.attr != "reset_parameters" else "reset"
        value_expr = self.visit(attr.value)
        return f"{value_expr}.{attr.attr}"

    ##############
    # Subscripting
    ##############
    def visit_Subscript(self, subscript):
        return f"_mpic_{subscript.value.attr}_{subscript.slice.value}"

    ############
    # Statements
    ############
    def visit_Assign(self, assign):
        target = assign.targets[0]  # only 1 target supported in the current grammar!
        target_expr = self.visit(target)

        if self.expr_attrs[assign.value] is LinearAttrs:
            linear_input_dim = self.visit(assign.value.args[0])
            linear_output_dim = self.visit(assign.value.args[1])
            self.write(f"_mpic_{target_expr}input_dim_ = {linear_input_dim};")
            self.write(f"_mpic_{target_expr}output_dim_ = {linear_output_dim};")
        else:
            value_expr = self.visit(assign.value)
            self.write(f"{target_expr} = {value_expr};")

    def visit_Pass(self, _):
        pass

    ##############
    # Control Flow
    ##############
    def visit_With(self, with_node):
        for stmt in with_node.body:
            self.visit(stmt)

    ################################
    # Function and Class Definitions
    ################################
    def visit_FunctionDef(self, func):
        for child in func.body:
            # XXX: cleaner to throw and handle an exception
            stmt = ast.unparse(child)
            if stmt.startswith("super(mpi.Module, self).__init__("):
                continue
            self.visit(child)

    def visit_Return(self, returns):
        return_expr = self.visit(returns.value)
        self.write(f"return {return_expr};")

    def generic_visit(self, node):
        raise RuntimeError(f"Internal error!: {astpp.dump(node)}")


def generate_func_body(
    classes: dict[str, Attrs],
    classdef: ast.ClassDef,
    expr_attrs,
    input_dim: str,
    output_dim: str,
) -> dict[str, str]:
    func_body = dict()
    for child in classdef.body:
        if isinstance(child, ast.FunctionDef):
            conf_input_dim = None
            conf_output_dim = None
            if child.name == "__init__":
                conf_input_dim = input_dim
                conf_output_dim = output_dim
            func_generator = FunctionGenerator(
                classes,
                classes[classdef.name],
                expr_attrs,
                conf_input_dim,
                conf_output_dim,
            )
            func_generator.visit(child)
            func_body[child.name] = func_generator.statements
    return func_body


def extract_helper_funcs(func_decls, local_vars, func_body):
    return [
        func_decl
        | {
            "local_vars": local_vars[func_decl["name"]],
            "body": func_body[func_decl["name"]],
        }
        for func_decl in func_decls
    ]
