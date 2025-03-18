from typing import Generic, TypeVar

# Type variables for keys and values
K = TypeVar("K")
V = TypeVar("V")


class frozendict(dict, Generic[K, V]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frozen = True  # Mark as frozen after initialization

    def __setitem__(self, key, value):
        self._raise_immutable_error()

    def __delitem__(self, key):
        self._raise_immutable_error()

    def update(self, *args, **kwargs):
        self._raise_immutable_error()

    def pop(self, *args, **kwargs):
        self._raise_immutable_error()

    def popitem(self, *args, **kwargs):
        self._raise_immutable_error()

    def clear(self):
        self._raise_immutable_error()

    def setdefault(self, *args, **kwargs):
        self._raise_immutable_error()

    def _raise_immutable_error(self):
        raise TypeError("frozendict is immutable and does not support modification")

    def __hash__(self):
        # Make it hashable like frozenset
        return hash(tuple(sorted(self.items())))
