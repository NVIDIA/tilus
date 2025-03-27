# pylint: disable=subprocess-run-check
import argparse
import importlib
import os
import pickle
import subprocess
import sys

_cuda_blocking_template = """
    CUDA_LAUNCH_BLOCKING=1 {python_executable} {python_script} {args} 
""".replace("\n", " ").strip()


def _cuda_blocking_run_func(script_path, func_name, args_pickled_path):
    with open(args_pickled_path, "rb") as f:
        args, kwargs = pickle.load(f)

    # remove the dir path of the current script from sys.path to avoid module overriding
    sys.path = [path for path in sys.path if not path.startswith(os.path.dirname(__file__))]

    try:
        sys.path.append(os.path.dirname(script_path))
        module = importlib.import_module(os.path.basename(script_path)[:-3])
    except Exception as e:
        raise RuntimeError("Can not import the python script: {}".format(script_path)) from e

    if not hasattr(module, func_name):
        raise RuntimeError('Can not find the function "{}" in {}'.format(func_name, script_path))

    func = getattr(module, func_name)

    try:
        func(*args, **kwargs)
    except Exception as e:
        raise RuntimeError('Error when running the function "{}"'.format(func_name)) from e


def cuda_blocking_run(func, *args, **kwargs):
    """
    Run the given function with CUDA_LAUNCH_BLOCKING=1 environment variable set.

    Usage:

    ```python
        from cuda_blocking_utils import cuda_blocking_run

        def func():
            # Some CUDA operations
            ...

        if __name__ == '__main__':
            # we need to wrap this part into '__main__' as this script will be imported in the utility script
            cuda_blocking_run(func)
    ```

    Parameters
    ----------
    func:
        The function to be executed with CUDA_LAUNCH_BLOCKING=1.

    args:
        The sequence of arguments to be passed to the function.

    kwargs:
        The dictionary of keyword arguments to be passed to the function.
    """
    import inspect
    import tempfile

    # get the python script path and function name
    script_path: str = inspect.getfile(func)
    func_name: str = func.__name__

    # dump args
    args_path: str = tempfile.mktemp() + ".pkl"
    with open(args_path, "wb") as f:
        pickle.dump((args, kwargs), f)

    command = _cuda_blocking_template.format(
        python_executable=sys.executable,
        python_script=__file__,
        args="{} {} {}".format(script_path, func_name, args_path),
    )
    command = " ".join(command.split())
    print("Running command: ")
    print(command)
    subprocess.run(command, shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script_path", type=str, help="the script path to the user's given func")
    parser.add_argument("func", type=str, help="the function to be executed")
    parser.add_argument("args", type=str, help="the arguments to be passed to the function (path to the pickled file)")
    args = parser.parse_args()
    _cuda_blocking_run_func(args.script_path, args.func, args.args)


if __name__ == "__main__":
    main()
