from typing import Sequence
from hidet.ir.module import IRModule


def merge_ir_modules(modules: Sequence[IRModule]) -> IRModule:
    if len(modules) == 0:
        return IRModule()
    merged = modules[0].copy()
    for module in modules[1:]:
        if module.namespace != merged.namespace:
            raise ValueError("Cannot merge IRModules with different namespaces")
        # merge global vars
        for name, var in module.global_vars.items():
            if name in merged.global_vars:
                raise ValueError("Global variable {} has already existed in module.".format(name))
            merged.global_vars[name] = var
        # merge functions
        for name, func in module.functions.items():
            if name in merged.functions:
                raise ValueError("Function {} has already existed in module.".format(name))
            merged.functions[name] = func
        # merge extern functions
        for name, var in module.extern_functions.items():
            if name in merged.extern_functions:
                continue
            merged.extern_functions[name] = var

        # merge include headers, include_dirs, linking_dirs, linking_libs, object_files
        merged.include_headers.extend(
            [header for header in module.include_headers if header not in merged.include_dirs]
        )
        merged.include_dirs.extend([dir for dir in module.include_dirs if dir not in merged.include_dirs])
        merged.linking_dirs.extend([dir for dir in module.linking_dirs if dir not in merged.linking_dirs])
        merged.linking_libs.extend([lib for lib in module.linking_libs if lib not in merged.linking_libs])
        merged.object_files.extend([file for file in module.object_files if file not in merged.object_files])

        # merge attrs
        # for key, value in module.attrs.items():
        #     if key in merged.attrs and merged.attrs[key] != value:
        #         raise ValueError("Attribute {} has already existed in module with a different value.".format(key))
        #     merged.attrs[key] = value

    return merged
