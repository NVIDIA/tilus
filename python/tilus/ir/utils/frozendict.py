from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class frozendict(dict, Generic[K, V]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frozen = True

    def __setitem__(self, key, value):
        if hasattr(self, "_frozen") and self._frozen:
            self._raise_immutable_error()
        else:
            super().__setitem__(key, value)

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
        return hash(tuple(sorted(self.items())))

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self._frozen = False
        super().__init__(state)
        self._frozen = True
