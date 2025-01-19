import torch


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    device = position_ids.device
    position_ids = position_ids.to(cos.device)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim).to(q.device)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim).to(q.device)
    # print("cos.shape", cos.device, "sin.shape", sin.device, "q.shape", q.device, "k.shape", k.device)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    position_ids = position_ids.to(device)
    return q_embed, k_embed
