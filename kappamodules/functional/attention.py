import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        backend="auto",
        **kwargs
):
    sdpa_kwargs = dict(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        **kwargs,
    )
    if backend == "auto":
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        return F.scaled_dot_product_attention(**sdpa_kwargs)
    if backend == "math":
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return F.scaled_dot_product_attention(**sdpa_kwargs)
    if backend in ["vanilla", "flops"]:
        assert not is_causal
        assert len(kwargs) == 0
        batch_size, _, _, head_dim = query.shape
        scale = head_dim ** -0.5
        qquery = query * scale
        attn = qquery @ key.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=dropout_p)
        x = attn @ value
        return x
    raise NotImplementedError(f"invalid backend {backend}")
