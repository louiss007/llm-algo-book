import sys

import pytest
import torch

sys.path.insert(0, ".")
from ch08 import ModelConfig, TinyLM
from tinylm.multimodal import (MultimodalTinyLM, VisionConfig, attention_multiplier,
                               contrastive_loss, image_token_count,
                               vq_loss, vq_quantize)


def toy_model(patch=3, projector="mlp", frozen=False):
    vcfg = VisionConfig(image_size=12, patch_size=patch, projector=projector)
    lm = TinyLM(ModelConfig(vocab_size=11, d_model=32, n_layers=1,
                            n_heads=4, d_ff=64, max_seq_len=64))
    model = MultimodalTinyLM(lm, vcfg, image_token_id=2)
    if frozen:
        for p in model.vision.parameters():
            p.requires_grad_(False)
    return model


def batch(m=16, n=4):
    ids = torch.tensor([[1, 3] + [2] * m for _ in range(n)])
    images = torch.rand(n, 3, 12, 12)
    labels = torch.full((n, m + 2), -100)
    labels[:, -1] = torch.randint(5, 11, (n,))
    return ids, images, labels


def test_image_token_count_and_merge():
    assert image_token_count(12, 3) == 16
    assert image_token_count(12, 3, 2) == 4
    assert image_token_count(336, 14) == 576
    assert image_token_count(336, 14, 2) == 144
    assert image_token_count(448, 14, 2) == 256
    with pytest.raises(AssertionError):
        image_token_count(10, 3)


def test_attention_multiplier():
    assert attention_multiplier(0, 10) == 1.0
    assert attention_multiplier(10, 10) == 4.0
    assert attention_multiplier(30, 10) == 16.0
    with pytest.raises(AssertionError):
        attention_multiplier(5, 0)


def test_patch_embed_equals_linear_patch():
    m = toy_model()
    x = torch.randn(2, 3, 12, 12)
    got = m.vision.patch(x)
    # 手动 unfold：[B, C, p, p, N] -> [B, N, C*p*p]
    patches = torch.nn.functional.unfold(x, kernel_size=3, stride=3)
    weight = m.vision.patch.proj.weight.reshape(32, -1)
    bias = m.vision.patch.proj.bias
    want = patches.transpose(1, 2) @ weight.t() + bias
    assert torch.allclose(got, want, atol=1e-6)


def test_vision_shapes_and_pool():
    for pool in ("tokens", "cls"):
        cfg = VisionConfig(image_size=12, patch_size=3, pool=pool)
        from tinylm.multimodal import VisionEncoder
        enc = VisionEncoder(cfg)
        out = enc(torch.randn(5, 3, 12, 12))
        assert out.shape == ((5, 16, 32) if pool == "tokens" else (5, 32))


def test_projector_kinds():
    for kind, expect in (("linear", 16), ("mlp", 16), ("resampler", 8)):
        m = toy_model(projector=kind)
        out = m.encode_image(torch.randn(3, 3, 12, 12))
        assert out.shape == (3, expect, 32)


def test_forward_uses_image():
    torch.manual_seed(0)
    m = toy_model()
    ids, images, _ = batch()
    with torch.no_grad():
        with_img = m(ids, images).logits
        no_img = m(ids).logits
    assert not torch.allclose(with_img, no_img)


def test_placeholder_count_must_match():
    m = toy_model()
    ids, images, _ = batch(m=16)
    ids[:, 2] = 3
    with pytest.raises(ValueError):
        m(ids, images)


def test_gradients_reach_projector_and_vision():
    m = toy_model()
    ids, images, labels = batch()
    out = m(ids, images, labels=labels)
    out.loss.backward()
    assert m.projector.net[0].weight.grad is not None
    assert m.vision.patch.proj.weight.grad is not None


def test_frozen_vision_has_no_grad():
    m = toy_model(frozen=True)
    ids, images, labels = batch()
    m(ids, images, labels=labels).loss.backward()
    assert m.vision.patch.proj.weight.grad is None
    assert m.projector.net[0].weight.grad is not None


def test_labels_only_last_position():
    m = toy_model()
    ids, images, labels = batch()
    out = m(ids, images, labels=labels)
    want = torch.nn.functional.cross_entropy(
        out.logits[:, -1], labels[:, -1])
    assert torch.allclose(out.loss, want, atol=1e-6)


def test_contrastive_loss_orders():
    torch.manual_seed(0)
    text = torch.randn(32, 16)
    image = 0.8 * text + 0.2 * torch.randn(32, 16)
    perm = torch.randperm(32)
    aligned = contrastive_loss(image, text, 0.1)
    shuffled = contrastive_loss(image, text[perm], 0.1)
    assert aligned < shuffled
    assert torch.allclose(contrastive_loss(image, text, 0.1),
                          contrastive_loss(text, image, 0.1))
    with pytest.raises(AssertionError):
        contrastive_loss(image, text, 0.0)


def test_vq_quantize_and_loss():
    codebook = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    z = torch.tensor([[0.1, -0.1], [1.9, 2.1], [0.9, 1.1]])
    zq, idx = vq_quantize(z, codebook)
    assert idx.tolist() == [0, 2, 1]
    assert torch.allclose(zq, codebook[idx])
    assert vq_loss(zq, zq).item() == 0.0
    with pytest.raises(RuntimeError):
        vq_quantize(torch.randn(4, 3), codebook)
