import cProfile
import os
import pstats
from datetime import datetime

REPORT_DIR = "./cprofile_reports"


def _ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


def _get_next_id():
    existing = [fname for fname in os.listdir(REPORT_DIR) if fname.endswith(".txt") or fname.endswith(".pstat")]
    ids = set()
    for fname in existing:
        try:
            base = os.path.splitext(fname)[0]
            ids.add(int(base.split("_")[0]))
        except (ValueError, IndexError):
            continue
    return max(ids, default=0) + 1


def cprofile_run(func, *args, warmup=1, repeat=1, **kwargs):
    """
    Profile the given function using cProfile, saving both .pstat and a human-readable .txt report.

    Args:
        func: The function to be profiled.
        *args: Positional arguments to pass to the function.
        warmup (int): Number of warmup runs (not profiled).
        repeat (int): Number of repeat runs (profiled and aggregated).
        **kwargs: Keyword arguments to pass to the function.
    """
    _ensure_report_dir()
    run_id = _get_next_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pstat_path = os.path.join(REPORT_DIR, f"{run_id}_{timestamp}.pstat")
    report_path = os.path.join(REPORT_DIR, f"{run_id}_{timestamp}.txt")

    # Warmup phase (no profiling)
    for _ in range(warmup):
        func(*args, **kwargs)

    # Profiling phase
    profiler = cProfile.Profile()
    profiler.enable()
    result = None
    for _ in range(repeat):
        result = func(*args, **kwargs)
    profiler.disable()

    profiler.dump_stats(pstat_path)

    with open(report_path, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats()

    print("[cProfile] Profiling complete. Reports saved to:")
    print(f"  - Raw stats: {pstat_path}")
    print(f"  - Human-readable: {report_path}")

    return result
