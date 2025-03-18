import shutil
from pathlib import Path
import tilus.option


def clear_cache(*items: str) -> None:
    """
    Clear the cache directory for the given items.

    Parameters
    ----------
    items: sequence[str]
        The path items append to the cache directory to determine the directory to clear.
    """
    root = Path(tilus.option.get_option("cache_dir")).resolve()
    dir_to_clear = root / Path(*items)
    print("Clearing tilus cache dir: {}".format(dir_to_clear))
    dir_to_clear.mkdir(parents=True, exist_ok=True)
    for item in dir_to_clear.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
