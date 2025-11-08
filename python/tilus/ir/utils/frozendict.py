# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

    def __hash__(self):  # type: ignore[override]
        return hash(tuple(sorted(self.items())))

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self._frozen = False
        super().__init__(state)
        self._frozen = True
