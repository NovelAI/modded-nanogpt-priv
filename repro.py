import torch
from torch import nn, Tensor
# import torch._inductor.config
# torch._inductor.config.triton.cudagraph_trees = False

device = torch.device("cuda")

hp_dtype=torch.bfloat16

class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(out_features=dim, in_features=dim, bias=False, dtype=hp_dtype)

    def forward(self, x: Tensor):
        y = self.proj(x)
        return y

dim=768

with torch.device('meta'):
    model = Model(dim=dim)
seed=42
gen=torch.Generator(device)
loss_fn = nn.MSELoss()

model.to_empty(device=device)
model.proj.weight.data.normal_(std=dim**-.5, generator=gen.manual_seed(seed))

model = torch.compile(model, dynamic=False, fullgraph=True, mode='reduce-overhead')

microbsz = 131072
target = torch.randn((1, microbsz, dim), device=device, dtype=hp_dtype, generator=gen.manual_seed(seed+2))

def with_cudagraph_do_fwdbwd(mod: Model):
    input = torch.randn((1, microbsz, dim), device=device, dtype=hp_dtype, generator=gen.manual_seed(seed+1), requires_grad=True)
    torch.compiler.cudagraph_mark_step_begin()
    out: Tensor = mod(input)
    loss: Tensor = loss_fn(out, target)
    loss.backward()

with_cudagraph_do_fwdbwd(model)
torch.cuda.synchronize()
model.zero_grad()
# gets past first fwdbwd just fine

# fails on .backward() of this subsequent invocation
with_cudagraph_do_fwdbwd(model)
torch.cuda.synchronize()