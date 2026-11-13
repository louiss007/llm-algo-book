"""第 25 章：多模态大模型的最小实现（tinylm/multimodal.py）。

只依赖 PyTorch。视觉塔、投影层与图文拼接都在这里；
``TinyLM.forward``、``CausalLMOutput`` 与 ``ModelConfig`` 不改。
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class VisionConfig:
    image_size: int = 12
    patch_size: int = 3
    in_channels: int = 3
    d_vision: int = 32
    n_layers: int = 2
    n_heads: int = 4
    mlp_ratio: float = 2.0
    pool: str = "tokens"          # tokens | cls
    projector: str = "mlp"        # linear | mlp | resampler
    n_queries: int = 8            # resampler 专用

    def __post_init__(self):
        assert self.image_size % self.patch_size == 0
        assert self.d_vision % self.n_heads == 0
        assert self.pool in {"tokens", "cls"}
        assert self.projector in {"linear", "mlp", "resampler"}
        if self.projector == "resampler":
            assert self.n_queries >= 1

def image_token_count(image_size, patch_size, merge=1):
    """(25.1) 与 (25.7)：切块并可选地做 ``merge x merge`` 合并。"""
    assert image_size % patch_size == 0
    n = (image_size // patch_size) ** 2
    assert n % (merge * merge) == 0
    return n // (merge * merge)

def attention_multiplier(n_image, n_text):
    """(25.3)：含图像 Token 的序列相对纯文本序列的注意力算力倍数。"""
    assert n_image >= 0 and n_text > 0
    return ((n_image + n_text) / n_text) ** 2

class PatchEmbed(nn.Module):
    """(25.4)：不重叠卷积等价于线性分块投影。"""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.proj = nn.Conv2d(cfg.in_channels, cfg.d_vision,
                              kernel_size=cfg.patch_size,
                              stride=cfg.patch_size)
        self.n_patches = (cfg.image_size // cfg.patch_size) ** 2

    def forward(self, pixel_values):
        z = self.proj(pixel_values)                 # (B, d_v, H/p, W/p)
        return z.flatten(2).transpose(1, 2)         # (B, N_p, d_v)

class VisionEncoder(nn.Module):
    """极简 ViT：分块嵌入 + CLS + 位置嵌入 + 若干 Pre-LN 块。"""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.patch = PatchEmbed(cfg)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_vision))
        self.pos = nn.Parameter(torch.zeros(1, self.patch.n_patches + 1, cfg.d_vision))
        h = cfg.n_heads
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.d_vision, nhead=h,
                dim_feedforward=int(cfg.d_vision * cfg.mlp_ratio),
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_vision)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, pixel_values):
        z = self.patch(pixel_values)
        B = z.shape[0]
        z = torch.cat([self.cls.expand(B, -1, -1), z], dim=1)
        z = z + self.pos
        for block in self.blocks:
            z = block(z)
        z = self.norm(z)
        return z[:, 0] if self.cfg.pool == "cls" else z[:, 1:]

class TokenResampler(nn.Module):
    """(25.6)：用可学习的 K 个查询做交叉注意力，把 N_p 个视觉 Token 压到 K 个。"""

    def __init__(self, d_vision, d_model, n_queries, n_heads=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_vision, d_model, bias=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model))
        self.ln_ff = nn.LayerNorm(d_model)

    def forward(self, z):
        B = z.shape[0]
        q = self.q_proj(self.queries).expand(B, -1, -1)
        kv = self.kv_proj(z)
        out, _ = self.attn(self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv),
                           need_weights=False)
        q = q + out
        return q + self.ff(self.ln_ff(q))

class Projector(nn.Module):
    """把视觉特征映射到语言模型的隐藏空间。"""

    def __init__(self, cfg: VisionConfig, d_model):
        super().__init__()
        if cfg.projector == "linear":
            self.net = nn.Linear(cfg.d_vision, d_model)
        elif cfg.projector == "mlp":
            self.net = nn.Sequential(
                nn.Linear(cfg.d_vision, 2 * cfg.d_vision), nn.GELU(),
                nn.Linear(2 * cfg.d_vision, d_model))
        else:
            self.net = TokenResampler(cfg.d_vision, d_model, cfg.n_queries,
                                      cfg.n_heads)

    def forward(self, z):
        if z.dim() == 2:
            z = z[:, None, :]
        return self.net(z)

class MultimodalTinyLM(nn.Module):
    """LLaVA 式拼接：图像 Token 占据文本中的占位位置，共用同一条因果序列。

    不改 ``TinyLM.forward``：用一个临时 forward hook 替换 ``tok_emb`` 的输出。
    """

    def __init__(self, lm, vision_cfg: VisionConfig, image_token_id: int):
        super().__init__()
        self.lm = lm
        self.vision_cfg = vision_cfg
        self.vision = VisionEncoder(vision_cfg)
        self.projector = Projector(vision_cfg, lm.cfg.d_model)
        self.image_token_id = image_token_id

    def encode_image(self, pixel_values):
        return self.projector(self.vision(pixel_values))   # (B, M, d_model)

    def forward(self, input_ids, pixel_values=None, *, labels=None,
                attention_mask=None):
        if pixel_values is None:
            return self.lm(input_ids, labels=labels, attention_mask=attention_mask)

        visual = self.encode_image(pixel_values)
        mask = input_ids == self.image_token_id
        if not torch.all(mask.sum(dim=1) == visual.shape[1]):
            raise ValueError("每行的图像占位符数量必须等于视觉 Token 数")
        emb = self.lm.tok_emb(input_ids).clone()
        emb[mask] = visual.reshape(-1, emb.shape[-1])      # (25.8)

        handle = self.lm.tok_emb.register_forward_hook(lambda m, i, o: emb)
        try:
            return self.lm(input_ids, labels=labels,
                           attention_mask=attention_mask)
        finally:
            handle.remove()

def contrastive_loss(image_emb, text_emb, temperature=0.07):
    """(25.5)：对称 InfoNCE，batch 内第 i 对为正样本。"""
    assert temperature > 0
    z_i = F.normalize(image_emb, dim=-1)
    z_t = F.normalize(text_emb, dim=-1)
    logits = z_i @ z_t.t() / temperature
    target = torch.arange(z_i.shape[0], device=z_i.device)
    return 0.5 * (F.cross_entropy(logits, target) +
                  F.cross_entropy(logits.t(), target))

def vq_quantize(z, codebook):
    """(25.9)：最近邻查表。``z`` 形状 (..., D)，``codebook`` 形状 (K, D)。"""
    flat = z.reshape(-1, z.shape[-1])
    dist = (flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(1).unsqueeze(0))
    idx = dist.argmin(dim=1)
    zq = codebook[idx].reshape_as(z)
    return zq, idx.reshape(z.shape[:-1])

def vq_loss(z, zq, beta=0.25):
    """(25.10)：码本损失 + 承诺损失。"""
    codebook_loss = F.mse_loss(zq, z.detach())
    commitment = F.mse_loss(z, zq.detach())
    return codebook_loss + beta * commitment
