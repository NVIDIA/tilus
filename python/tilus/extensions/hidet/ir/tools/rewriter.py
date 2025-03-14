from typing import Union, Mapping, TypeVar
from hidet.ir.node import Node
from hidet.ir.tools import rewrite as original_rewrite

_K = TypeVar("_K", bound=Node)
_V = TypeVar("_V", bound=Node)


def rewrite(node: Union[Node, tuple, list, dict], rewrite_map: Mapping[_K, _V], clone_internal_var=False):
    return original_rewrite(node, rewrite_map, clone_internal_var)  # type: ignore
