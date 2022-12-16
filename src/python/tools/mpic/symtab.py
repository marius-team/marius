from contextlib import contextmanager

from marius.tools.mpic.builtins import builtin_attrs
from marius.tools.mpic.utils import Attrs


class SymbolTable:
    def __init__(self, global_attrs):
        self.scopes = [builtin_attrs, global_attrs]
        self.graph_local_scope = 0

    def enter_scope(self):
        self.scopes.append(Attrs())

    def exit_scope(self):
        return self.scopes.pop()

    @contextmanager
    def managed_scope(self):
        self.enter_scope()
        yield
        self.exit_scope()

    def find_symbol(self, symbol: str):
        for scope in reversed(self.scopes):
            if symbol in scope:
                return scope[symbol]

        return None

    def add_symbol(self, symbol: str, value_attrs: Attrs):
        local_scope = self.scopes[-1]
        local_scope[symbol] = value_attrs
