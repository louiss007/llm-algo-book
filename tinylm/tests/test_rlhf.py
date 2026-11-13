"""第 16 章测试：Bradley-Terry、GAE、PPO 裁剪与 KL 估计。"""
import math

import pytest
import torch

from ch08 import ModelConfig, TinyLM
from tinylm.posttrain import ppo as P
from tinylm.posttrain import rm as RM

def tiny_cfg():
    return ModelConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=2,
                       d_ff=128, max_seq_len=32, pos_encoding="rope",
                       norm="rmsnorm", ffn="swiglu", bias=False)

def model(seed=0):
    torch.manual_seed(seed)
    return TinyLM(tiny_cfg())

def test_bt_loss_zero_margin_is_log_two():
    r = torch.zeros(4)
    loss = RM.bradley_terry_loss(r, r)
    assert torch.allclose(loss, torch.tensor(math.log(2.0)), atol=1e-6)

def test_bt_loss_decreases_with_margin():
    win = torch.tensor([0.0, 1.0, 2.0, 4.0])
    lose = torch.zeros(4)
    losses = [float(RM.bradley_terry_loss(win[:i + 1], lose[:i + 1]))
              for i in range(4)]
    assert losses[0] > losses[1] > losses[2] > losses[3]

def test_preference_probability_is_symmetric():
    a = torch.tensor([0.3, -1.2, 2.0])
    b = torch.tensor([-0.4, 0.5, 0.1])
    p = RM.preference_probability(a, b)
    q = RM.preference_probability(b, a)
    assert torch.allclose(p + q, torch.ones_like(p), atol=1e-6)

def test_scalar_model_shapes_and_score():
    cfg = tiny_cfg()
    reward = RM.ScalarModel(cfg)
    ids = torch.randint(0, 16, (3, 7))
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[2, 5:] = False
    assert reward(ids, mask).shape == (3, 7)
    assert reward.score(ids, mask).shape == (3,)
    # score 取最后一个有效位置
    values = reward(ids, mask)
    last = mask.long().sum(-1) - 1
    assert torch.allclose(reward.score(ids, mask),
                          values.gather(1, last[:, None]).squeeze(1))
    assert torch.allclose(reward.score(ids, mask)[2], values[2, 4])

def test_init_scalar_keeps_backbone_weights():
    cfg = tiny_cfg()
    lm = model()
    scalar, dropped = RM.init_scalar_from_lm(cfg, lm.state_dict(), seed=1)
    assert dropped == ["lm_head.weight"]
    for (name, p), q in zip(lm.named_parameters(), scalar.backbone.parameters()):
        if name.startswith("lm_head"):
            continue
        assert torch.equal(p, q), name

def test_gae_hand_example():
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.zeros(1, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    adv, ret = P.compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.ones(1, 3))
    assert torch.allclose(ret, torch.ones(1, 3))
    adv0, _ = P.compute_gae(rewards, values, mask, gamma=1.0, lam=0.0)
    assert torch.allclose(adv0, torch.tensor([[0.0, 0.0, 1.0]]))

def test_gae_ignores_padding():
    rewards = torch.tensor([[0.0, 1.0, 0.0]])
    values = torch.zeros(1, 3)
    mask = torch.tensor([[True, True, False]])
    adv, ret = P.compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert adv[0, 2] == 0.0 and ret[0, 2] == 0.0
    assert torch.allclose(adv[0, :2], torch.ones(2))

def test_clipped_surrogate_three_regions():
    ratio = torch.tensor([0.5, 1.0, 2.0])
    adv = torch.tensor([-1.0, 1.0, 1.0])
    out = P.clipped_surrogate(ratio, adv, 0.2)
    assert torch.allclose(out, torch.tensor([-0.8, 1.0, 1.2]))

def test_ratio_is_one_when_policy_unchanged():
    m = model()
    ids = torch.randint(0, 16, (2, 9))
    mask = torch.ones(2, 9, dtype=torch.bool)
    logp = P.token_logprobs(m, ids, mask)
    ratio = torch.exp(logp - logp)
    assert torch.allclose(ratio, torch.ones_like(ratio))

def test_kl_estimators_zero_for_identical_models():
    m = model()
    ids = torch.randint(0, 16, (2, 9))
    mask = torch.ones(2, 9, dtype=torch.bool)
    with torch.no_grad():
        logp = P.token_logprobs(m, ids, mask)
        assert float(P.kl_penalty(logp, logp, mask)) == 0.0
        k3, per_seq = P.kl_estimator_k3(logp, logp, mask)
        assert float(k3) == 0.0 and torch.allclose(per_seq, torch.zeros(2))

def test_sampling_respects_horizon_and_pads_after_eos():
    cfg = P.RLConfig(max_new=4, temperature=1.0)
    m = model()
    prompts = torch.tensor([[1, 2, 5, 3], [1, 2, 6, 3]])
    gen = torch.Generator().manual_seed(0)
    seqs = P.sample_responses(m, prompts, cfg, gen)
    assert seqs.size(1) <= prompts.size(1) + cfg.max_new
    for row in seqs:
        body = row[prompts.size(1):]
        eos = (body == cfg.eos_id).nonzero()
        if eos.numel():
            first = int(eos[0])
            assert bool((body[first + 1:] == cfg.pad_id).all())
    mask = P.response_mask(seqs, prompts.size(1), cfg.pad_id)
    assert not bool(mask[:, :prompts.size(1)].any())
    assert bool(mask[:, prompts.size(1):].any())

def test_rollout_shapes_and_finite():
    cfg = P.RLConfig(max_new=5, epochs=1)
    policy = model(0)
    ref = model(0)
    value = RM.ScalarModel(tiny_cfg())
    reward = RM.ScalarModel(tiny_cfg())
    prompts = torch.tensor([[1, 2, 5, 3], [1, 2, 6, 3], [1, 2, 7, 3]])
    gen = torch.Generator().manual_seed(1)
    batch = P.collect_rollout(policy, ref, value, reward, prompts, cfg, gen)
    assert batch.seqs.shape == batch.attn.shape == batch.mask.shape
    for tensor in (batch.logp, batch.values, batch.rewards,
                   batch.adv, batch.returns):
        assert torch.isfinite(tensor).all()
    assert batch.rm_score.shape == (3,)
    # 策略与参考同权重时 k1 KL 应为 0
    with torch.no_grad():
        logp_pi = P.token_logprobs(policy, batch.seqs, batch.attn)
        logp_ref = P.token_logprobs(ref, batch.seqs, batch.attn)
        assert float(P.kl_penalty(logp_pi, logp_ref, batch.mask)) == 0.0

def test_rlconfig_validates_clip_and_beta():
    with pytest.raises(AssertionError):
        P.RLConfig(beta=-0.1)
    with pytest.raises(AssertionError):
        P.RLConfig(clip=1.0)
    with pytest.raises(AssertionError):
        P.RLConfig(max_new=0)
