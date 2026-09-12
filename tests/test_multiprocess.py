# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import tilus.utils.multiprocess as module


@pytest.mark.parametrize("method", ["parallel_imap", "parallel_map"])
@pytest.mark.parametrize("workers, expected", [(None, 4), (192, 4), (2, 2)])
def test_worker_count_is_bounded_by_jobs(monkeypatch, method, workers, expected):
    sizes = []

    class Pool:
        def __init__(self, count):
            sizes.append(count)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def imap(self, func, indices):
            return map(func, indices)

        def map(self, func, indices, chunksize):
            return list(map(func, indices))

    monkeypatch.setattr("tilus.option.get_option", lambda name: 192)
    monkeypatch.setattr(module.multiprocessing, "get_context", lambda method: SimpleNamespace(Pool=Pool))

    result = list(getattr(module, method)(lambda value: value * 2, [0, 1, 2, 3], num_workers=workers))

    assert result == [0, 2, 4, 6]
    assert sizes == [expected]
    assert module._job_queue is None


@pytest.mark.parametrize("method", ["parallel_imap", "parallel_map"])
def test_empty_jobs_start_no_pool(monkeypatch, method):
    def unexpected_context(*args):
        pytest.fail("Empty jobs should not start workers")

    monkeypatch.setattr(module.multiprocessing, "get_context", unexpected_context)
    assert list(getattr(module, method)(lambda value: value, [])) == []
