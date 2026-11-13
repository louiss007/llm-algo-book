"""有缓存与无缓存的 logits 等价性。"""
import torch

from tinylm.config import ModelConfig
from tinylm.inference.kv_cache import KVCache, cache_bytes, cached_forward
from tinylm.model import TinyLM


def make_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4,
                      n_kv_heads=2, d_ff=64, max_seq_len=32, attention="gqa",
                      norm="rmsnorm", ffn="swiglu", pos_encoding="rope",
                      bias=False, tie_embeddings=False)
    return TinyLM(cfg).eval(), cfg


def test_cached_decode_matches_full_forward():
    model, cfg = make_model()
    ids = torch.randint(0, 32, (1, 12))
    with torch.no_grad():
        full = model(ids).logits
        cache = KVCache(cfg, batch_size=1, max_len=32)
        pieces = [cached_forward(model, ids[:, :5], cache)]
        for t in range(5, 12):
            pieces.append(cached_forward(model, ids[:, t:t + 1], cache))
    cached = torch.cat(pieces, dim=1)
    assert cached.shape == full.shape
    assert (cached - full).abs().max().item() < 1e-4


def test_cache_bytes_formula():
    _, cfg = make_model()
    d_head = cfg.d_model // cfg.n_heads
    expected = cfg.n_layers * 4 * (2 * cfg.n_kv_heads * d_head) * 2
    assert cache_bytes(cfg, batch_size=1, seq_len=4, b_elem=2) == expected
