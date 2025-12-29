from typing import Literal
import torch
from torch import Tensor

# based on k-diffusion's _apply_rotary_emb_inplace,
# MIT-licensed, by Katherine Crowson
# https://github.com/crowsonkb/k-diffusion/blob/21d12c91ad4550e8fcf3308ff9fe7116b3f19a08/k_diffusion/models/image_transformer_v2.py#L188C5-L188C30
def _rope_inplace(t: Tensor, cos: Tensor, sin: Tensor, coeff: Literal[1, -1]) -> None:
    ty, tx = t.unbind(-2)
    ty_roped = ty.mul(cos).addcmul_(sin, tx, value=coeff)
    tx_roped = tx.mul(cos).addcmul_(sin, ty, value=-coeff)
    ty.copy_(ty_roped)
    tx.copy_(tx_roped)


# based on k-diffusion's ApplyRotaryEmbeddingInplace,
# MIT-licensed, by Katherine Crowson
# https://github.com/crowsonkb/k-diffusion/blob/21d12c91ad4550e8fcf3308ff9fe7116b3f19a08/k_diffusion/models/image_transformer_v2.py#L202
class RopeInPlace(torch.autograd.Function):
    @staticmethod
    def forward(t: Tensor, cos: Tensor, sin: Tensor, coeff: Literal[1, -1] = 1):
        "NOTE: passing the coeff arg -1 explicitly gave better compiled performance than relying on arg defaulting"
        _rope_inplace(t, cos, sin, coeff=coeff)
        return t

    @staticmethod
    def setup_context(ctx, inputs, output):
        t, cos, sin, coeff = inputs
        # mark t as dirty because we will modify it in-place. this ensures backward() will be called.
        ctx.mark_dirty(t)
        ctx.save_for_backward(cos, sin)
        ctx.coeff = coeff

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        # clone made because we must "NEVER" modify grad-w.r.t-input in-place.
        # https://pytorch.org/docs/main/notes/extending.html#how-to-use
        # https://discuss.pytorch.org/t/is-it-safe-to-modify-outputs-grad-and-return-as-inputs-grad/201630
        # it seemed to work for simple cases (including ours as far as we can tell), just being cautious really.
        # this copy seems to slow down the train step by 0.4% (non-compiled) / 0.7% (compiled)
        grad_output = RopeInPlace.apply(grad_output.clone(), cos, sin, -ctx.coeff)
        return grad_output, None, None, None