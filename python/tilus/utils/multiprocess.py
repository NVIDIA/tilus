import multiprocessing
from typing import Any, Callable, Iterable, Optional, Sequence


class JobQueue:
    def __init__(self, func: Callable[[Any], Any], jobs: Sequence[Any] = tuple()):
        self.func: Callable = func
        self.jobs: Sequence[Any] = jobs


_job_queue: Optional[JobQueue] = None


def _wrapped_func(job_index):
    """
    Wrapper function for parallel_imap.

    We use this function to avoid pickling the jobs.
    """
    assert _job_queue and job_index < len(_job_queue.jobs)

    job = _job_queue.jobs[job_index]
    func = _job_queue.func

    return func(job)


def parallel_imap(func: Callable, jobs: Sequence[Any], num_workers: Optional[int] = None) -> Iterable[Any]:
    if num_workers == 1:
        yield from map(func, jobs)
        return

    global _job_queue

    if _job_queue is not None:
        raise RuntimeError("Cannot call parallel_map recursively.")

    _job_queue = JobQueue(func, jobs)

    try:
        if num_workers is None:
            from tilus.option import get_option

            num_workers = get_option("parallel_workers")

        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(num_workers) as pool:
            yield from pool.imap(_wrapped_func, range(len(jobs)))
    finally:
        _job_queue = None


def parallel_map(func: Callable, jobs: Sequence[Any], num_workers: Optional[int] = None) -> Iterable[Any]:
    global _job_queue

    if _job_queue is not None:
        raise RuntimeError("Cannot call parallel_map recursively.")

    _job_queue = JobQueue(func, jobs)

    try:
        if num_workers is None:
            from tilus.option import get_option

            num_workers = get_option("parallel_workers")

        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(num_workers) as pool:
            ret = pool.map(_wrapped_func, range(len(jobs)), chunksize=1)

        _job_queue = None
        return ret
    finally:
        _job_queue = None
