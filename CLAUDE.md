## Cache

Tilus caches generated kernels. During development, set the cache directory via `tilus.option.cache_dir("some-cache-dir")` to inspect the generated `source.cu` files.

### Cache structure
- **`/scripts/`** — Each script is a template with a tuning space. Contains references to its instantiated programs.
- **`/programs/`** — Each program is a concrete instantiation of a script with a specific schedule.

### Tips
- The cache directory can be safely deleted at any time.
- To inspect the CUDA kernel generated for a specific script, delete the cache directory, run the program, and check the newly generated `source.cu`.
- Use `debug_schedule=dict(...)` to pin a specific schedule, so only that single configuration is compiled.
