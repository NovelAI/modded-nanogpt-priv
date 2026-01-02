# benchmark based on Carson Poole's RMSNorm gist
# https://gist.github.com/carsonpo/5011a284d54a6c7c872da4b991d449ab
# with changes by Alex Birch to support profiling, optional compress_py kernel, try out cudagraphs, prefer assert_close and prefer do_bench
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import torch
from functools import partial
from torch.testing import assert_close
from torch.profiler import ProfilerActivity, profile
from torch.nn.functional import rms_norm
try:
    import compress_py
    cpy_found = True
except ImportError:
    print("[WARN] compress_py not available on your system")
    cpy_found = False

#---
# do_bench is from triton, MIT-licensed
# https://github.com/triton-lang/triton/blob/11ec6354/python/triton/testing.py#L127
# with fixes by Alex Birch to clear grads before warmup steps too (cudagraphs models are sensitive to tensor reuse)
# (in case you wanted to test backwards pass after this)

from triton import runtime
from triton.testing import _summarize_statistics

def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Default is "mean".
    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = runtime.driver.active.get_device_interface()

    if grad_to_none is not None:
        for x in grad_to_none:
            x.grad = None
    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        runtime.driver.active.clear_cache(cache)
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        fn()
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)
#---

def rmsnorm_eager(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x_fp32 = x.float()
    rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + 1e-5)
    return (x_fp32 * rms * weight).type_as(x)


def rmsnorm_custom(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return compress_py.rmsnorm_forward(x, weight, 1e-5)


@torch.compile(mode="max-autotune", dynamic=False, fullgraph=True)
def compiled_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x_fp32 = x.float()
    rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + 1e-5)
    return (x_fp32 * rms * weight).type_as(x)

@torch.compile(mode="reduce-overhead", dynamic=False, fullgraph=True)
def cudagraph_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x_fp32 = x.float()
    rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + 1e-5)
    return (x_fp32 * rms * weight).type_as(x)

def rmsnorm_builtin_eager(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return rms_norm(x.float(), normalized_shape=(x.shape[-1],), weight=weight, eps=1e-5).type_as(x)

@torch.compile(mode="max-autotune", dynamic=False, fullgraph=True)
def rmsnorm_builtin_compiled(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return rms_norm(x.float(), normalized_shape=(x.shape[-1],), weight=weight, eps=1e-5).type_as(x)

@torch.compile(mode="reduce-overhead", dynamic=False, fullgraph=True)
def rmsnorm_builtin_cudagraph(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return rms_norm(x.float(), normalized_shape=(x.shape[-1],), weight=weight, eps=1e-5).type_as(x)

# I know it's not so beautiful that it requires a niladic function instead of passing arguments
# but CPU overhead matters in a microbenchmark, and python 3.10 doesn't have a JIT, so I'll avoid doing arg-spreading in case it matters
def with_step_begin(fn):
    def better_fn():
        torch.compiler.cudagraph_mark_step_begin()
        return fn()
    return better_fn

@dataclass
class BenchResult:
    ms_per_iter: float

    @property
    def iter_per_s(self) -> float:
        return 1000 / self.ms_per_iter

class StrategyName(Enum):
    Eager = 'eager'
    Compiled = 'compiled'
    CudaGraph = 'cudagraph'
    Custom = 'custom'
    BuiltinEager = 'builtin-eager'
    BuiltinCompiled = 'builtin-compiled'
    BuiltinCudaGraph = 'builtin-cudagraph'

def main():
    device = torch.device('cuda')
    gen = torch.Generator(device)
    x = torch.randn((1024, 512), generator=gen.manual_seed(42), device=device).type(torch.float8_e4m3fn)
    weight = torch.randn((512), generator=gen.manual_seed(43), device=device)

    if check_correctness := False:
        eager_out = rmsnorm_eager(x, weight)
        eager_out_float = eager_out.float()

        builtin_out = rmsnorm_builtin_eager(x, weight)
        assert_close(eager_out_float, builtin_out.float())
        print("builtin_out is close to eager_out")

        builtin_compiled_out = rmsnorm_builtin_compiled(x, weight)
        assert_close(eager_out_float, builtin_compiled_out.float())
        print("builtin_compiled_out is close to eager_out")

        builtin_cudagraph_out = with_step_begin(partial(rmsnorm_builtin_cudagraph, x, weight))()
        assert_close(eager_out_float, builtin_cudagraph_out.float())
        print("builtin_cudagraph_out is close to eager_out")

        compiled_out = compiled_rmsnorm(x, weight)
        assert_close(eager_out_float, compiled_out.float())
        print("compiled_out is close to eager_out")

        cudagraph_out = with_step_begin(partial(cudagraph_rmsnorm, x, weight))()
        assert_close(eager_out_float, cudagraph_out.float())
        print("cudagraph_out is close to eager_out")

        if cpy_found:
            custom_out = rmsnorm_custom(x, weight)
            assert_close(eager_out_float, custom_out.float())
            print("custom_out is close to eager_out")

    if do_profile := False:
        wait, warmup, active = 1, 1, 1
        prof_its = wait + warmup + active
        prof = profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            # stack traces introduce sufficient CPU overhead as to mislead, so don't believe such profiles entirely.
            # with_stack=True,
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
        )
        torch.cuda.synchronize()
        for fn, label in zip((
            partial(rmsnorm_eager, x, weight),
            partial(compiled_rmsnorm, x, weight),
            with_step_begin(partial(cudagraph_rmsnorm, x, weight)),
            partial(rmsnorm_builtin_eager, x, weight),
            partial(rmsnorm_builtin_compiled, x, weight),
            with_step_begin(partial(rmsnorm_builtin_cudagraph, x, weight)),
            *(partial(rmsnorm_custom, x, weight),) * cpy_found,
        ), (
            StrategyName.Eager,
            StrategyName.Compiled,
            StrategyName.CudaGraph,
            StrategyName.BuiltinEager,
            StrategyName.BuiltinCompiled,
            StrategyName.BuiltinCudaGraph,
            *(StrategyName.Custom,) * cpy_found,
        ), strict=True):
            with prof:
                for _ in range(prof_its):
                    fn()
                    torch.cuda.synchronize()
                    prof.step()
            trace_dir = Path("out_trace_rmsnorm")
            trace_dir.mkdir(exist_ok=True)
            profile_path = trace_dir / f"{label.value}.json"
            print(f"Saving profile to {profile_path}")
            prof.export_chrome_trace(str(profile_path))

    if do_benchmark := True:
        warmup, rep = 1000, 2000
        bench_results: dict[StrategyName, BenchResult] = {}
        bench_results[StrategyName.Eager] = BenchResult(do_bench(partial(rmsnorm_eager, x, weight), rep=rep, warmup=warmup))
        bench_results[StrategyName.Compiled] = BenchResult(do_bench(partial(compiled_rmsnorm, x, weight), rep=rep, warmup=warmup))
        bench_results[StrategyName.CudaGraph] = BenchResult(do_bench(with_step_begin(partial(cudagraph_rmsnorm, x, weight)), rep=rep, warmup=warmup))
        bench_results[StrategyName.BuiltinEager] = BenchResult(do_bench(partial(rmsnorm_builtin_eager, x, weight), rep=rep, warmup=warmup))
        bench_results[StrategyName.BuiltinCompiled] = BenchResult(do_bench(partial(rmsnorm_builtin_compiled, x, weight), rep=rep, warmup=warmup))
        bench_results[StrategyName.BuiltinCudaGraph] = BenchResult(do_bench(with_step_begin(partial(rmsnorm_builtin_cudagraph, x, weight)), rep=rep, warmup=warmup))

        if cpy_found:
            bench_results[StrategyName.Custom] = BenchResult(do_bench(partial(rmsnorm_custom, x, weight), rep=rep, warmup=warmup))

        bench_result = "\n".join((f"{name.value.rjust(17)}:  {result.ms_per_iter:5.3f}ms  {result.iter_per_s/1000:6.2f}kit/s" for name, result in bench_results.items()))
        print(bench_result)


if __name__ == "__main__":
    main()
