import os
import time
import gc
from dataclasses import dataclass
from typing import Any, Dict

import jax
import jax.numpy as jnp
from jax import tree_util

# Optional memory measurement (host RSS)
try:
    import psutil
except Exception:
    psutil = None

# -------------------------
# Pytree definition
# -------------------------
@dataclass(frozen=True)
class MyTree:
    a: Any
    b: Any
    c: Any

def _tree_flatten(t: MyTree):
    children = (t.a, t.b, t.c)
    aux_data = None
    return children, aux_data

def _tree_unflatten(aux_data, children):
    a, b, c = children
    return MyTree(a=a, b=b, c=c)

tree_util.register_pytree_node(MyTree, _tree_flatten, _tree_unflatten)

# -------------------------
# Helpers
# -------------------------
def rss_bytes() -> int:
    if psutil is None:
        return -1
    return psutil.Process(os.getpid()).memory_info().rss

def fmt_bytes(n: int) -> str:
    if n < 0:
        return "n/a"
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"

def block_until_ready(x):
    # Handle pytrees
    leaves, _ = tree_util.tree_flatten(x)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return x

def now():
    return time.perf_counter()

def time_it(fn, iters: int) -> float:
    t0 = now()
    for _ in range(iters):
        y = fn()
    block_until_ready(y)
    return now() - t0

def clear_everything():
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass

def compilation_cache_size_bytes() -> int:
    try:
        return jax.profiler.compilation_cache_size()
    except Exception:
        return -1

# -------------------------
# Data sizes
# -------------------------
# Tune these so you don't OOM; defaults are "large-ish" but should run on most machines.
N = int(os.environ.get("BENCH_N", "1048576"))  # ~4 MiB per float32 vector
DTYPE = jnp.float32

# For vmap: number of batch items for x
BATCH = int(os.environ.get("BENCH_BATCH", "64"))

# Iterations (after compile)
ITERS = int(os.environ.get("BENCH_ITERS", "200"))

# -------------------------
# Build pytree and inputs
# -------------------------
key = jax.random.key(0)

a = jax.random.normal(key, (N,), dtype=DTYPE)
b = jax.random.normal(jax.random.key(1), (N,), dtype=DTYPE)
c = jax.random.normal(jax.random.key(2), (N,), dtype=DTYPE)

TREE_GLOBAL = MyTree(a=a, b=b, c=c)

x = jax.random.normal(jax.random.key(3), (N,), dtype=DTYPE)
x_batched = jax.random.normal(jax.random.key(4), (BATCH, N), dtype=DTYPE)

# -------------------------
# Workloads (multiple jitted functions)
# -------------------------
@jax.jit
def g1_global(x):
    return x * TREE_GLOBAL.a + TREE_GLOBAL.b

@jax.jit
def g2_global(x):
    return jnp.tanh(x + TREE_GLOBAL.c)

@jax.jit
def g3_global(x):
    y = g1_global(x)
    z = g2_global(x)
    return (y + z).mean()

@jax.jit
def g1_arg(tree, x):
    return x * tree.a + tree.b

@jax.jit
def g2_arg(tree, x):
    return jnp.tanh(x + tree.c)

@jax.jit
def g3_arg(tree, x):
    y = g1_arg(tree, x)
    z = g2_arg(tree, x)
    return (y + z).mean()

# vmap over x (batched input)
g3_global_vmap_x = jax.jit(jax.vmap(g3_global))
g3_arg_vmap_x = jax.jit(jax.vmap(lambda xb: g3_arg(TREE_GLOBAL, xb)))

# -------------------------
# Benchmark runner
# -------------------------
def bench_case(name: str, compile_call, run_call) -> Dict[str, object]:
    clear_everything()

    rss0 = rss_bytes()
    cc0 = compilation_cache_size_bytes()

    # Compile (first call)
    t_compile0 = now()
    y = compile_call()
    block_until_ready(y)
    t_compile = now() - t_compile0

    rss1 = rss_bytes()
    cc1 = compilation_cache_size_bytes()

    # Warm-up a bit more
    for _ in range(5):
        y = run_call()
    block_until_ready(y)

    rss2 = rss_bytes()
    t_run = time_it(run_call, ITERS)
    rss3 = rss_bytes()

    return {
        "name": name,
        "compile_s": t_compile,
        "run_s_total": t_run,
        "run_s_per_iter": t_run / ITERS,
        "rss_delta_compile": (rss1 - rss0) if (rss0 >= 0 and rss1 >= 0) else -1,
        "rss_delta_run": (rss3 - rss2) if (rss2 >= 0 and rss3 >= 0) else -1,
        "compile_cache_delta": (cc1 - cc0) if (cc0 >= 0 and cc1 >= 0) else -1,
    }

def print_result(r: Dict[str, object]):
    print(f"\n== {r['name']} ==")
    print(f"compile time:          {r['compile_s']:.6f} s")
    print(f"run total ({ITERS}x):        {r['run_s_total']:.6f} s")
    print(f"run per iter:          {float(r['run_s_per_iter'])*1e6:.2f} us")
    print(f"RSS delta compile:     {fmt_bytes(int(r['rss_delta_compile']))}")
    print(f"RSS delta run:         {fmt_bytes(int(r['rss_delta_run']))}")
    print(f"compile cache delta:   {fmt_bytes(int(r['compile_cache_delta']))}")

def main():
    print("JAX:", jax.__version__)
    print("Backend:", jax.default_backend())
    print("Device:", jax.devices()[0])
    print("N:", N, "dtype:", DTYPE, "BATCH:", BATCH, "ITERS:", ITERS)
    if psutil is None:
        print("psutil not installed -> RSS memory measurements are n/a. Install with: pip install psutil")

    results = []

    # Non-vmap
    results.append(
        bench_case(
            "global capture (g3_global(x))",
            compile_call=lambda: g3_global(x),
            run_call=lambda: g3_global(x),
        )
    )
    results.append(
        bench_case(
            "pass arg (g3_arg(tree, x))",
            compile_call=lambda: g3_arg(TREE_GLOBAL, x),
            run_call=lambda: g3_arg(TREE_GLOBAL, x),
        )
    )
    # vmap over x
    results.append(
        bench_case(
            "vmap over x, global capture (jit(vmap(g3_global))(x_batched))",
            compile_call=lambda: g3_global_vmap_x(x_batched),
            run_call=lambda: g3_global_vmap_x(x_batched),
        )
    )
    results.append(
        bench_case(
            "vmap over x, pass arg (tree fixed, jit(vmap(...))(x_batched))",
            compile_call=lambda: g3_arg_vmap_x(x_batched),
            run_call=lambda: g3_arg_vmap_x(x_batched),
        )
    )

    for r in results:
        print_result(r)

if __name__ == "__main__":
    main()