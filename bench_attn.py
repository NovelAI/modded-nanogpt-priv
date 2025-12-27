import math
from dataclasses import dataclass
from functools import partial
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from kernels import get_kernel
from triton.testing import do_bench
from torch.testing import assert_close

try:
    if dont_bother_community_kernel := True:
        raise ValueError("skipping community kernel load")
    flash_attn_interface = get_kernel('varunneal/flash-attention-3').flash_attn_interface
    get_attn_out = lambda out: out

except Exception as e:
    print(f"[WARN] failed to load kernel. Error was: <{e}>. Will try falling back to own flash_attn_interface.")
    import flash_attn_interface
    def get_attn_out(out):
        assert isinstance(out, tuple)
        out, lse = out
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
            theta.cos().to(torch.bfloat16), persistent=False
        )
        self.sin = nn.Buffer(
            theta.sin().to(torch.bfloat16), persistent=False
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
        if use_builtin := False:
            y = torch.ops.aten._flash_attention_forward(q[0], k[0], v[0], cum_seq_q=seqlens, cum_seq_k=seqlens,
                                                            max_q=max_len, max_k=max_len, dropout_p=0.0,
                                                            is_causal=True, return_debug_mask=False, scale=attn_scale, window_size_left=bm_size, window_size_right=0)
            y, _, _, _, _ = y
        else:
            y = flash_attn_interface.flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
                                                            max_seqlen_q=max_len, max_seqlen_k=max_len,
                                                            causal=True, softmax_scale=attn_scale, window_size=(bm_size, 0))
            # some versions of flash_attn_varlen_func return (out, lse) instead of out
            y = get_attn_out(y)
        # y = q
        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, sa_lambdas[1] * self.qkvo_w[self.dim * 3:].type_as(y))  # sa_lambdas[1] pre-multiplied to O @shenberg
        return y

class CausalSelfAttentionNext(CausalSelfAttentionOrig): ...

head_dim=128
with torch.device('cuda'):
    yarn = Yarn(head_dim, max_seq_len)

hp_dtype=torch.bfloat16
dim=768
# seqlens=torch.tensor((0, args.train_max_seq_len), dtype=torch.int32, device=device)
avg_seqlen=400 # median doc length is ~400
minibsz=args.train_bs_schedule[0]
pergpu_minibsz=minibsz//grad_accum_steps
microbsz = pergpu_minibsz // world_size
max_num_docs = next_multiple_of_n(microbsz // 300, n=128)
seqlens=torch.arange(0, avg_seqlen*max_num_docs, avg_seqlen, dtype=torch.int32, device=device).clamp_max_(microbsz)
ve = torch.randn((microbsz, dim), device=device, dtype=hp_dtype, requires_grad=True)
sa_lambdas = torch.tensor((.5, 1.), device=device, requires_grad=True)

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
seed=42
gen=torch.Generator(device)
loss_fn = nn.MSELoss()

for attn in (orig, next):
    attn.to_empty(device=device)
    attn.qkvo_w.data.normal_(std=attn.dim**-.5, generator=gen.manual_seed(seed))
    attn.attn_gate.weight.data.normal_(std=attn.attn_gate.in_features**-.5, generator=gen.manual_seed(seed))

orig.forward = torch.compile(orig.forward, dynamic=False, fullgraph=True)
# orig.forward = torch.compile(orig.forward)

input = torch.randn((1, microbsz, dim), device=device, dtype=hp_dtype, generator=gen.manual_seed(seed+1), requires_grad=True)
target = torch.randn((1, microbsz, dim), device=device, dtype=hp_dtype, generator=gen.manual_seed(seed+2))

def do_fwd(mod: CausalSelfAttentionBase):
    attn_args = AttnArgs(
        ve=ve.clone(),
        sa_lambdas=sa_lambdas.clone(),
        seqlens=seqlens,
        bm_size=short_bm,
        cos=yarn.cos,
        sin=yarn.sin,
        attn_scale=yarn.attn_scale,
        # attn_scale=head_dim**-.5,
        key_shift=False
    )
    out: Tensor = mod(input.clone(), attn_args=attn_args)
    return out

def do_lossbwd(out: Tensor):
    loss: Tensor = loss_fn(out, target)
    loss.backward()
    return loss

def do_fwdbwd(mod: CausalSelfAttentionBase):
    out: Tensor = do_fwd(mod)
    do_lossbwd(out)

if test_correctness := False:
    out_orig = do_fwd(orig)
    out_next = do_fwd(next)
    assert_close(out_orig, out_next)


    do_lossbwd(out_orig)
    do_lossbwd(out_next)
    assert_close(orig.qkvo_w.grad, next.qkvo_w.grad)
    # K and V grads don't match
    # assert_close(orig.qkvo_w.grad.split(768)[1], next.qkvo_w.grad.split(768)[1])
    # assert_close(orig.qkvo_w.grad.split(768)[2], next.qkvo_w.grad.split(768)[2])
    assert_close(orig.attn_gate.weight.grad, next.attn_gate.weight.grad)

if test_latency := True:
    orig_ms: float = do_bench(partial(do_fwdbwd, mod=orig))
    next_ms: float = do_bench(partial(do_fwdbwd, mod=next))
    orig_its: float = 1000 / orig_ms
    next_its: float = 1000 / next_ms
    print(f"""
orig: {orig_its:.2f} it/s
next: {next_its:.2f} it/s
""")
pass