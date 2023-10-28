import re
from dataclasses import dataclass

ClassType = str
Attrs = dict


@dataclass
class Arg:
    name: str
    attrs: Attrs


@dataclass
class Callable:
    args: list[Arg]
    return_attrs: Attrs


# TODO: Should we inherit from TypeError?
# TODO: Show context of compiler error instead of stack trace!
class CompileError(Exception):
    """
    Base class for errors in marius script
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class SynError(CompileError):
    """
    Syntax errors for scripts that do not follow the Grammar
    """

    pass


class SemError(CompileError):
    """
    Semantic errors for scripts that fail type checking
    """

    pass


def camel_to_snake(name):
    """
    See https://stackoverflow.com/a/1176023/12160191
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
