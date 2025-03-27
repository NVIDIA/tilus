import json
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any

from tilus.option import get_option

# Module-level storage for statistics
durations_wallclock: dict[str, list[float]] = defaultdict(list)  # Wall-clock times
durations_perf: dict[str, list[float]] = defaultdict(list)  # Performance counter times
call_counts: dict[str, int] = defaultdict(int)  # Call counts


def collect_duration(name=None):
    """Decorator to collect wall-clock and perf_counter execution times, plus call count."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Wall-clock time
            start_wall = time.time()
            # Performance counter time
            start_perf = time.perf_counter()

            result = func(*args, **kwargs)

            end_wall = time.time()
            end_perf = time.perf_counter()

            elapsed_wall = end_wall - start_wall
            elapsed_perf = end_perf - start_perf

            func_name = name if name is not None else func.__name__
            durations_wallclock[func_name].append(elapsed_wall)
            durations_perf[func_name].append(elapsed_perf)
            call_counts[func_name] += 1
            return result

        return wrapper

    return decorator


def collect_times(name=None):
    """Decorator to collect only the number of calls."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name if name is not None else func.__name__
            call_counts[func_name] += 1
            return func(*args, **kwargs)

        return wrapper

    return decorator


def summary():
    """Print a formatted summary of collected statistics."""
    output = []
    output.append("\n=== Statistics Summary ===")
    output.append(
        f"{'Function':<20} {'Calls':<8} {'Avg Wall (s)':<15} {'Total Wall (s)':<15} {'Avg Perf (s)':<15} {'Total Perf (s)':<15}"
    )
    output.append("-" * 88)

    # Combine all tracked functions
    all_functions = set(durations_wallclock.keys()) | set(durations_perf.keys()) | set(call_counts.keys())

    for func_name in sorted(all_functions):
        calls = call_counts[func_name]
        if func_name in durations_wallclock:
            wall_times = durations_wallclock[func_name]
            perf_times = durations_perf[func_name]
            avg_wall = sum(wall_times) / len(wall_times) if wall_times else 0
            total_wall = sum(wall_times)
            avg_perf = sum(perf_times) / len(perf_times) if perf_times else 0
            total_perf = sum(perf_times)
            output.append(
                f"{func_name:<20} {calls:<8} {avg_wall:<15.4f} {total_wall:<15.4f} {avg_perf:<15.4f} {total_perf:<15.4f}"
            )
        else:
            output.append(f"{func_name:<20} {calls:<8} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    output.append("=========================\n")

    return "\n".join(output)


def save_stats():
    """Save run stats and update total stats at exit."""
    try:
        cache_dir = Path(get_option("cache_dir"))
        stats_dir = cache_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        # Determine the next run ID
        run_files = list(stats_dir.glob("run_*.txt"))
        run_ids = [int(f.stem.split("_")[1]) for f in run_files if f.stem.startswith("run_")]
        next_id = max(run_ids, default=-1) + 1

        # Write run-specific summary
        run_file = stats_dir / f"run_{next_id}.txt"
        with run_file.open("w") as f:
            f.write(summary())

        # Load existing total stats from JSON
        total_json = stats_dir / "total.json"
        total_data: dict[str, dict[str, Any]] = {
            "durations_wallclock": {},
            "durations_perf": {},
            "call_counts": {},
        }
        if total_json.exists():
            with total_json.open("r") as f:
                total_data = json.load(f)

        # Update total stats
        for func_name in durations_wallclock:
            total_data["durations_wallclock"].setdefault(func_name, []).extend(durations_wallclock[func_name])
            total_data["durations_perf"].setdefault(func_name, []).extend(durations_perf[func_name])
        for func_name in call_counts:
            total_data["call_counts"][func_name] = total_data["call_counts"].get(func_name, 0) + call_counts[func_name]

        # Save updated total stats to JSON
        with total_json.open("w") as f:
            json.dump(total_data, f, indent=2)

        # Write total summary to total.txt
        total_wallclock = defaultdict(list, total_data["durations_wallclock"])
        total_perf = defaultdict(list, total_data["durations_perf"])
        total_counts = defaultdict(int, total_data["call_counts"])

        output = []
        output.append("\n=== Total Statistics Across All Runs ===")
        output.append(
            f"{'Function':<20} {'Calls':<8} {'Avg Wall (s)':<15} {'Total Wall (s)':<15} {'Avg Perf (s)':<15} {'Total Perf (s)':<15}"
        )
        output.append("-" * 88)

        all_functions = set(total_wallclock.keys()) | set(total_perf.keys()) | set(total_counts.keys())
        for func_name in sorted(all_functions):
            calls = total_counts[func_name]
            if func_name in total_wallclock:
                wall_times = total_wallclock[func_name]
                perf_times = total_perf[func_name]
                avg_wall = sum(wall_times) / len(wall_times) if wall_times else 0
                total_wall = sum(wall_times)
                avg_perf = sum(perf_times) / len(perf_times) if perf_times else 0
                total_perf = sum(perf_times)
                output.append(
                    f"{func_name:<20} {calls:<8} {avg_wall:<15.4f} {total_wall:<15.4f} {avg_perf:<15.4f} {total_perf:<15.4f}"
                )
            else:
                output.append(f"{func_name:<20} {calls:<8} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        output.append("=======================================\n")

        total_file = stats_dir / "total.txt"
        with total_file.open("w") as f:
            f.write("\n".join(output))

    except Exception as e:
        # Silent fail to avoid crashing the program at exit
        print(f"Failed to save stats: {e}")


# # Register the save_stats function to run at exit
# atexit.register(save_stats)
