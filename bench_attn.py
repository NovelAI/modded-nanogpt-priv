from typing import Callable, Optional
import math
from pathlib import Path
from dataclasses import dataclass
from functools import partial
import torch
from torch import nn, Tensor, FloatTensor, IntTensor
import torch.nn.functional as F
# from triton.testing import do_bench
from torch.testing import assert_close
from torch.profiler import ProfilerActivity, profile
from src.rope import RopeInPlace
from src.do_bench import do_bench
# import torch._inductor.config
# torch._inductor.config.triton.cudagraph_trees = False
# torch._inductor.config.triton.cudagraphs = True

import torch._dynamo
# import logging
torch._dynamo.config.verbose = True

if use_fa3 := True:
    get_attn_out: Callable[[Tensor|tuple[Tensor, ...]], Tensor]
    if use_local_fa3_kernel := True:
        from kernels import get_local_kernel
        flash_attn_interface = get_local_kernel(repo_path=Path('hf-kernels/flash-attention-3'), package_name='flash_attention_3').flash_attn_interface
        get_attn_out = lambda out: out
    elif use_community_fa3_kernel := False:
        from kernels import get_kernel
        flash_attn_interface = get_kernel('varunneal/flash-attention-3').flash_attn_interface
        get_attn_out = lambda out: out
    elif use_dist_fa3 := False:
        import flash_attn_interface
        def get_attn_out(out):
            assert isinstance(out, tuple)
            out, lse = out
            return out
    else:
        raise ValueError("well you have to pick one")
    def varlen_attn(
        q: FloatTensor,
        k: FloatTensor,
        v: FloatTensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        cum_seq_q: Optional[IntTensor] = None,
        cum_seq_k: Optional[IntTensor] = None,
        causal=False,
        scale: Optional[float] = None,
        window_size_left: Optional[int] = None,
        window_size_right: Optional[int] = None,
    ) -> FloatTensor:
        assert (window_size_left is None) == (window_size_right is None)
        window_size = None if window_size_left is None else (window_size_left, window_size_right)
        out = flash_attn_interface.flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cum_seq_q,
            cu_seqlens_k=cum_seq_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            softmax_scale=scale,
            window_size=window_size,
        )
        return get_attn_out(out)
else:
    print("[WARN] falling back to torch builtin private varlen attn API, which is probably FA2")
    def varlen_attn(
        q: FloatTensor,
        k: FloatTensor,
        v: FloatTensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        cum_seq_q: Optional[IntTensor] = None,
        cum_seq_k: Optional[IntTensor] = None,
        causal=False,
        scale: Optional[float] = None,
        window_size_left: Optional[int] = None,
        window_size_right: Optional[int] = None,
    ) -> FloatTensor:
        out, _, _, _, _ = torch.ops.aten._flash_attention_forward(
            q, k, v,
            cum_seq_q=cum_seq_q,
            cum_seq_k=cum_seq_k,
            max_q=max_seqlen_q,
            max_k=max_seqlen_k,
            dropout_p=0.0,
            is_causal=causal,
            return_debug_mask=False,
            scale=scale,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        return out



def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

@dataclass
class Hyperparameters:
    train_bs_schedule: tuple = (8 * 2048 * 8, 16 * 2048 * 8, 24 * 2048 * 8)
    train_max_seq_len: int = 128 * 16
    val_batch_size: int = 4 * 64 * 1024 * 8

args = Hyperparameters()

rank = 0
world_size = 8
grad_accum_steps = 8 // world_size
max_seq_len=args.val_batch_size // (grad_accum_steps * world_size)

device = torch.device("cuda", 0)
torch.cuda.set_device(device)

@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float
    key_shift: bool

hp_dtype=torch.bfloat16
# eps: float = torch.finfo(torch.bfloat16).eps # rms_norm defaults to x.dtype, though it upcasts x to at least f32 before checking its dtype
def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(
            theta.cos().to(hp_dtype), persistent=False
        )
        self.sin = nn.Buffer(
            theta.sin().to(hp_dtype), persistent=False
        )
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos.copy_(theta.cos())
        self.sin.copy_(theta.sin())
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = (
        cos[None, : x_BTHD.size(-3), None, :],
        sin[None, : x_BTHD.size(-3), None, :],
    )
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()  # @Grad62304977 and others

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

class CausalSelfAttentionBase(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.call_super_init = True
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim

        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        std = self.dim ** -0.5
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKVO weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        # Simplified layout by @chrisjmccormick
        self.qkvo_w = nn.Parameter(torch.empty(self.dim * 4, self.hdim))
        # label all modules for explicit optimizer grouping
        self.qkvo_w.label = 'attn'

        with torch.no_grad():
            self.qkvo_w[:self.dim * 3].uniform_(-bound, bound)  # init QKV weights
            self.qkvo_w[self.dim * 3:].zero_()  # init O weights to zero

        # sparse gated attention to enable context based no-op by @classiclarryd
        self.attn_gate = CastedLinear(12, num_heads)
        self.attn_gate.weight.label = 'attn_gate'

class CausalSelfAttentionOrig(CausalSelfAttentionBase):
    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas, key_shift = attn_args.ve, attn_args.sa_lambdas, attn_args.key_shift
        seqlens, attn_scale, bm_size = attn_args.seqlens, attn_args.attn_scale, attn_args.bm_size

        q, k, v = F.linear(x, sa_lambdas[0] * self.qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if key_shift:
            # shift keys forward for the stationary head dims. Enables 1-layer induction.
            k[:, 1:, :, self.head_dim//4:self.head_dim//2] = k[:, :-1, :, self.head_dim//4:self.head_dim//2]
            k[:, 1:, :, self.head_dim//4+self.head_dim//2:] = k[:, :-1, :, self.head_dim//4+self.head_dim//2:]
        if ve is not None:
            v = v + ve.view_as(v) # @ KoszarskyB & @Grad62304977

        max_len = args.train_max_seq_len if self.training else (args.val_batch_size // (grad_accum_steps * world_size))

        # use flash_attn over flex_attn @varunneal. flash_attn_varlen suggested by @YouJiacheng
        y: Tensor = varlen_attn(
            q[0],
            k[0],
            v[0],
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
            cum_seq_q=seqlens,
            cum_seq_k=seqlens,
            causal=True,
            scale=attn_scale,
            window_size_left=bm_size,
            window_size_right=0,
        )
        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, sa_lambdas[1] * self.qkvo_w[self.dim * 3:].type_as(y))  # sa_lambdas[1] pre-multiplied to O @shenberg
        return y

class CausalSelfAttentionCG(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim

        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        self.qkv = nn.Linear(out_features=self.dim * 3, in_features=self.hdim, bias=False)
        self.o = nn.Linear(out_features=self.dim, in_features=self.hdim, bias=False)

        # sparse gated attention to enable context based no-op by @classiclarryd
        self.attn_gate = CastedLinear(12, num_heads)
        self.attn_gate.weight.label = 'attn_gate'

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas, key_shift = attn_args.ve, attn_args.sa_lambdas, attn_args.key_shift
        seqlens, attn_scale, bm_size = attn_args.seqlens, attn_args.attn_scale, attn_args.bm_size

        q, k, v = F.linear(x, sa_lambdas[0] * self.qkv.weight.type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if key_shift:
            # shift keys forward for the stationary head dims. Enables 1-layer induction.
            k[:, 1:, :, self.head_dim//4:self.head_dim//2] = k[:, :-1, :, self.head_dim//4:self.head_dim//2]
            k[:, 1:, :, self.head_dim//4+self.head_dim//2:] = k[:, :-1, :, self.head_dim//4+self.head_dim//2:]
        if ve is not None:
            v = v + ve.view_as(v) # @ KoszarskyB & @Grad62304977

        max_len = args.train_max_seq_len if self.training else (args.val_batch_size // (grad_accum_steps * world_size))

        # use flash_attn over flex_attn @varunneal. flash_attn_varlen suggested by @YouJiacheng
        y: Tensor = varlen_attn(
            q[0],
            k[0],
            v[0],
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
            cum_seq_q=seqlens,
            cum_seq_k=seqlens,
            causal=True,
            scale=attn_scale,
            window_size_left=bm_size,
            window_size_right=0,
        )
        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, sa_lambdas[1] * self.o.weight.type_as(y))  # sa_lambdas[1] pre-multiplied to O @shenberg
        return y

class CausalSelfAttentionNext(CausalSelfAttentionOrig):
    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas, key_shift = attn_args.ve, attn_args.sa_lambdas, attn_args.key_shift
        seqlens, attn_scale, bm_size = attn_args.seqlens, attn_args.attn_scale, attn_args.bm_size

        # q, k, v = F.linear(x, sa_lambdas[0] * self.qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        qkv = F.linear(x, sa_lambdas[0] * self.qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim)
        # q, k, v = qkv.chunk(3, dim=-2)
        # q, k = norm(q), norm(k) # QK norm @Grad62304977
        # qk = qkv[..., :2 * self.num_heads, :]
        qk, v = qkv.tensor_split((2*self.num_heads,), dim=-2)
        qk = norm(qk)
        # qk = rotary(qk, cos, sin)
        RopeInPlace.apply(
            qk.unflatten(-1, (2, -1)),
            cos[:qk.size(-3), None],
            sin[:qk.size(-3), None],
            1,
        )
        q,k=qk.chunk(2, dim=-2)
        # q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if key_shift:
            # shift keys forward for the stationary head dims. Enables 1-layer induction.
            k[:, 1:, :, self.head_dim//4:self.head_dim//2] = k[:, :-1, :, self.head_dim//4:self.head_dim//2]
            k[:, 1:, :, self.head_dim//4+self.head_dim//2:] = k[:, :-1, :, self.head_dim//4+self.head_dim//2:]
        if ve is not None:
            v = v + ve.view_as(v) # @ KoszarskyB & @Grad62304977

        max_len = args.train_max_seq_len if self.training else (args.val_batch_size // (grad_accum_steps * world_size))

        # use flash_attn over flex_attn @varunneal. flash_attn_varlen suggested by @YouJiacheng
        y: Tensor = varlen_attn(
            q[0],
            k[0],
            v[0],
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
            cum_seq_q=seqlens,
            cum_seq_k=seqlens,
            causal=True,
            scale=attn_scale,
            window_size_left=bm_size,
            window_size_right=0,
        )
        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, sa_lambdas[1] * self.qkvo_w[self.dim * 3:].type_as(y))  # sa_lambdas[1] pre-multiplied to O @shenberg
        return y

head_dim=128
with torch.device('cuda'):
    yarn = Yarn(head_dim, max_seq_len)

dim=768
# seqlens=torch.tensor((0, args.train_max_seq_len), dtype=torch.int32, device=device)
avg_seqlen=400 # median doc length is ~400
minibsz=args.train_bs_schedule[0]
pergpu_minibsz=minibsz//grad_accum_steps
microbsz = pergpu_minibsz // world_size
max_num_docs = next_multiple_of_n(microbsz // 300, n=128)
seqlens=torch.arange(0, avg_seqlen*max_num_docs, avg_seqlen, dtype=torch.int32, device=device).clamp_max_(microbsz)

short_bm=128

num_heads=6
with torch.device('meta'):
    orig = CausalSelfAttentionOrig(
        dim=dim,
        head_dim=head_dim,
        num_heads=num_heads,
    )
    next = CausalSelfAttentionNext(
        dim=dim,
        head_dim=head_dim,
        num_heads=num_heads,
    )
    cg = CausalSelfAttentionCG(
        dim=dim,
        head_dim=head_dim,
        num_heads=num_heads,
    )
seed=42
gen=torch.Generator(device)
loss_fn = nn.MSELoss()

for attn in (orig, next, cg):
    attn.to_empty(device=device)
    if attn is not cg:
        attn.qkvo_w.data.normal_(std=attn.dim**-.5, generator=gen.manual_seed(seed))
    attn.attn_gate.weight.data.normal_(std=attn.attn_gate.in_features**-.5, generator=gen.manual_seed(seed))

qkv, o = orig.qkvo_w.tensor_split((orig.dim*3,), dim=-2)
cg.qkv.weight.data.copy_(qkv)
cg.o.weight.data.copy_(o)

cg_grads_to_none = [cg.qkv.weight, cg.o.weight, cg.attn_gate.weight]
next_grads_to_none = [next.qkvo_w, next.attn_gate.weight]

# cg_opts = {
#     "backend": "inductor",
#     "options": {
#         "triton.cudagraphs": True
#     }
# }

orig = torch.compile(orig, dynamic=False, fullgraph=True)

# import torch._inductor.config as triton_config
# triton_config.triton.cudagraphs = True
cg = torch.compile(cg, dynamic=False, fullgraph=True, mode='reduce-overhead')
next = torch.compile(next, dynamic=False, fullgraph=True, mode='reduce-overhead')

ve = torch.randn((microbsz, dim), device=device, dtype=hp_dtype, requires_grad=True)
sa_lambdas = torch.tensor((.5, 1.), device=device, requires_grad=True)

input = torch.randn((1, microbsz, dim), device=device, dtype=hp_dtype, generator=gen.manual_seed(seed+1), requires_grad=True)
target = torch.randn((1, microbsz, dim), device=device, dtype=hp_dtype, generator=gen.manual_seed(seed+2))

def do_fwd(mod: CausalSelfAttentionBase):
    attn_args = AttnArgs(
        ve=ve,
        sa_lambdas=sa_lambdas,
        seqlens=seqlens,
        bm_size=short_bm,
        cos=yarn.cos,
        sin=yarn.sin,
        attn_scale=yarn.attn_scale,
        # attn_scale=head_dim**-.5,
        key_shift=False
    )
    out: Tensor = mod(input, attn_args=attn_args)
    return out

def do_lossbwd(out: Tensor):
    loss: Tensor = loss_fn(out, target)
    loss.backward()
    return loss

def do_fwdbwd(mod: nn.Module):
    out: Tensor = do_fwd(mod)
    do_lossbwd(out)

def with_cudagraph_do_fwd(mod: nn.Module):
    torch.compiler.cudagraph_mark_step_begin()
    return do_fwd(mod)

def with_cudagraph_do_fwdbwd(mod: nn.Module):
    torch.compiler.cudagraph_mark_step_begin()
    out: Tensor = do_fwd(mod)
    loss: Tensor = loss_fn(out, target)
    loss.backward()
    return loss

if test_correctness := False:
    out_orig = do_fwd(orig)
    out_next = do_fwd(next)
    # rtol=1e-2 due to torch.finfo(torch.bfloat16).resolution
    assert_close(out_orig, out_next, rtol=1e-2, atol=1e-2)
    # assert_close(out_orig, out_next)


    do_lossbwd(out_orig)
    do_lossbwd(out_next)
    assert_close(orig.qkvo_w.grad, next.qkvo_w.grad, rtol=1e-6, atol=5e-5)
    assert_close(orig.attn_gate.weight.grad, next.attn_gate.weight.grad)

inputs_to_none = [input, ve, sa_lambdas]
def clear_input_grads():
    for t in inputs_to_none:
        t.grad = None

if do_profile := True:
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
    for mod, label, wants_cudagraph in zip((orig, next, cg), ("orig", "next", "cg"), (False, True, True), strict=True):
        do_fwdbwd_ = with_cudagraph_do_fwdbwd if wants_cudagraph else do_fwdbwd
        with prof:
            for step in range(prof_its):
                clear_input_grads()
                mod.zero_grad()
                do_fwdbwd_(mod)
                torch.cuda.synchronize()
                prof.step()
        trace_dir = Path("out_trace_nanogpt")
        trace_dir.mkdir(exist_ok=True)
        profile_path = trace_dir / f"{label}.json"
        print(f"Saving profile to {profile_path}")
        prof.export_chrome_trace(str(profile_path))

if test_fwdbwd := False:
    clear_input_grads()
    cg.zero_grad()
    with_cudagraph_do_fwdbwd(cg)
    torch.cuda.synchronize()
    # import torch._inductor.config as config
    # print(config.triton.cudagraphs)  # Should be True
    clear_input_grads()
    cg.zero_grad()
    with_cudagraph_do_fwdbwd(cg)
    torch.cuda.synchronize()

if test_latency := False:
    warmup, rep = 1000, 2000
    orig_ms: float = do_bench(partial(do_fwdbwd, mod=orig), rep=rep, warmup=warmup, grad_to_none=inputs_to_none)
    cg_ms: float = do_bench(partial(with_cudagraph_do_fwdbwd, mod=cg), rep=rep, warmup=warmup, grad_to_none=[*cg_grads_to_none, *inputs_to_none])
    next_ms: float = do_bench(partial(with_cudagraph_do_fwdbwd, mod=next), rep=rep, warmup=warmup, grad_to_none=[*next_grads_to_none, *inputs_to_none])
    orig_its: float = 1000 / orig_ms
    cg_its: float = 1000 / cg_ms
    next_its: float = 1000 / next_ms
    print(f"""
orig: {orig_its:.2f} it/s
  cg: {cg_its:.2f} it/s
next: {next_its:.2f} it/s
""")
pass