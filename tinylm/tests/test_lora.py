"""第 15 章测试：LoRA 等价性、参数账本与 NF4 量化。"""
import pytest
import torch

from ch08 import ModelConfig, TinyLM
from tinylm.posttrain import lora as L

VOCAB = 16

def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2,
                      d_ff=128, max_seq_len=32, pos_encoding="rope",
                      norm="rmsnorm", ffn="swiglu", bias=False)
    return TinyLM(cfg)

def ids(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (2, 6), generator=g)

def test_initial_output_identity():
    model = tiny_model()
    x = ids()
    ref = model(x).logits
    L.apply_lora(model, L.LoraConfig(r=4))
    out = model(x).logits
    assert torch.equal(ref, out)

def test_merge_equivalence_and_unmerge():
    model = tiny_model()
    x = ids()
    L.apply_lora(model, L.LoraConfig(r=4, alpha=8))
    g = torch.Generator().manual_seed(3)
    for _, p in L.lora_parameters(model):
        p.data.normal_(0.0, 0.02, generator=g)
    model.eval()
    ref = model(x).logits
    linears = [m for m in model.modules() if isinstance(m, L.LoRALinear)]
    for m in linears:
        m.merge()
    merged = model(x).logits
    for m in linears:
        m.unmerge()
    back = model(x).logits
    assert torch.allclose(ref, merged, atol=1e-6)
    assert torch.allclose(ref, back, atol=1e-7)

def test_only_adapter_parameters_trainable():
    model = tiny_model()
    cfg = L.LoraConfig(r=4, alpha=8)
    expected = sum(n for _, n in L.lora_module_params(model, cfg))
    L.apply_lora(model, cfg)
    stats = L.trainable_stats(model)
    assert stats["trainable"] == expected
    assert stats["ratio"] < 0.25
    for name, p in model.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        assert p.requires_grad == (leaf in ("lora_A", "lora_B"))

def test_zero_init_b_blocks_first_step_on_a():
    model = tiny_model()
    L.apply_lora(model, L.LoraConfig(r=4))
    out = model(ids(), labels=ids(1))
    out.loss.backward()
    a_grads, b_grads = [], []
    for name, p in model.named_parameters():
        if name.endswith("lora_A"):
            a_grads.append(p.grad.abs().max())
        if name.endswith("lora_B"):
            b_grads.append(p.grad.abs().max())
    assert max(b_grads) > 0
    assert max(a_grads) == 0

def test_adapter_state_roundtrip():
    model = tiny_model()
    L.apply_lora(model, L.LoraConfig(r=4))
    g = torch.Generator().manual_seed(5)
    for _, p in L.lora_parameters(model):
        p.data.normal_(0.0, 0.02, generator=g)
    state = L.lora_state_dict(model)
    ref = model(ids()).logits
    for _, p in L.lora_parameters(model):
        p.data.zero_()
    assert not torch.allclose(ref, model(ids()).logits)
    L.load_lora_state_dict(model, state)
    assert torch.allclose(ref, model(ids()).logits)

def test_target_selection():
    model = tiny_model()
    L.apply_lora(model, L.LoraConfig(r=2, target_modules=("q_proj", "v_proj")))
    names = [n for n, m in model.named_modules() if isinstance(m, L.LoRALinear)]
    assert names and all(n.endswith(("q_proj", "v_proj")) for n in names)

def test_nf4_roundtrip_is_close():
    torch.manual_seed(0)
    w = torch.randn(64, 96)
    deq, codes, absmax = L.nf4_quantize(w, block_size=64)
    err = (w - deq).norm() / w.norm()
    assert codes.max() <= 15 and codes.numel() == w.numel()
    assert absmax.numel() == 96
    assert err < 0.2

def test_nf4_storage_bits():
    plain = L.nf4_storage_bytes(1 << 20, double_quant=False)
    double = L.nf4_storage_bytes(1 << 20, double_quant=True)
    assert 4.4 < plain["bits_per_param"] < 4.6
    assert 4.1 < double["bits_per_param"] < 4.2

def test_qlora_keeps_base_frozen():
    model = tiny_model()
    L.apply_lora(model, L.LoraConfig(r=4))
    info = L.quantize_lora_bases(model)
    assert info["bits_per_param"] < 4.3
    for m in model.modules():
        if isinstance(m, L.LoRALinear):
            assert not m.base.weight.requires_grad
            assert m.lora_A.requires_grad
    out = model(ids(), labels=ids(1))
    out.loss.backward()
    assert any(p.grad is not None for _, p in L.lora_parameters(model))

def test_bad_adapter_keys_raise():
    model = tiny_model()
    L.apply_lora(model, L.LoraConfig(r=2))
    with pytest.raises(KeyError):
        L.load_lora_state_dict(model, {})
