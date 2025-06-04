import importlib
from typing import Callable, TypeVar

T = TypeVar("T")


def update(obj_full_qualname: str) -> Callable[[T], T]:
    """
    Update the object defined with the given full qualified name with the decorated object.


    ```python
    from tilus.extensions.utils import update

    @update("hidet.ir.tools.type_infer.infer_type")
    @update("hidet.ir.tools.infer_type")
    def infer_type(expr):
        ...
    ```

    Parameters
    ----------
    obj_full_qualname: str
        The full qualified name of the object to update, e.g. "hidet.ir.tools.type_infer.infer_type".

    Returns
    -------
    ret:
        The updated object.
    """
    # split the full qualified name into module and object name
    module_name, obj_name = obj_full_qualname.rsplit(".", 1)

    # import the module to check if the object exists
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Module {module_name} does not exist, cannot update {obj_name}.") from e

    if not hasattr(module, obj_name):
        raise AttributeError(f"Module {module_name} does not have an object named {obj_name}, cannot update.")

    def decorator(new_obj):
        """
        Decorator to update the object with the given full qualified name.
        """
        setattr(module, obj_name, new_obj)
        return new_obj

    return decorator
