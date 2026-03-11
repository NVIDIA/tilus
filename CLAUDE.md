## Cache

Tilus caches generated kernels. During development, set the cache directory via `tilus.option.cache_dir("some-cache-dir")` to inspect the generated `source.cu` files.

### Cache structure
- **`/scripts/`** — Each script is a template with a tuning space. Contains references to its instantiated programs.
- **`/programs/`** — Each program is a concrete instantiation of a script with a specific schedule.

### Tips
- The cache directory can be safely deleted at any time.
- To inspect the CUDA kernel generated for a specific script, delete the cache directory, run the program, and check the newly generated `source.cu`.
- Use `debug_schedule=dict(...)` to pin a specific schedule, so only that single configuration is compiled.

## Debug Tips

### Proxy fence required between `store_shared` and `tma.shared_to_global`

`self.store_shared(...)` writes to shared memory via the **generic proxy**, while `self.tma.shared_to_global(...)` reads from shared memory via the **async proxy**. A `fence.proxy.async.shared::cta` is required between them to ensure the generic proxy writes are visible to the async proxy. Without this fence, the TMA engine may read stale data.

```python
    ...
    self.store_shared(s_c, ...)
    self.fence.async_view(space="shared")   # fence.proxy.async.shared::cta
    self.sync()
    with self.single_thread():
        self.tma.shared_to_global(s_c, g_c, ...)
        self.tma.commit_group()
        self.tma.wait_group(n=0)
    ...
```
