"""第 16 章：rollout、GAE 与 PPO 裁剪目标（tinylm/posttrain/ppo.py）。"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class RLConfig:
    """PPO 训练脚本参数，不进入 ``ModelConfig``。"""

    lr: float = 1e-3
    beta: float = 0.5               # KL 正则系数
    clip: float = 0.2               # PPO 裁剪系数
    gamma: float = 1.0              # 语言模型回合是有限期，折扣取 1
    lam: float = 0.95               # GAE 的偏差—方差折中
    epochs: int = 2                 # 每批 rollout 重复更新次数
    minibatch: int = 16
    vf_coef: float = 0.5
    max_new: int = 6                # 响应最多生成多少个 Token（含 EOS）
    temperature: float = 1.0
    grad_clip: float = 1.0
    normalize_reward: bool = True
    pad_id: int = 0
    eos_id: int = 4

    def __post_init__(self):
        assert self.beta >= 0.0
        assert 0.0 < self.clip < 1.0
        assert self.epochs >= 1
        assert self.max_new >= 1

@dataclass
class Rollout:
    seqs: torch.Tensor        # (B, T)
    attn: torch.Tensor        # (B, T) bool
    mask: torch.Tensor        # (B, T) bool，响应 Token（含 EOS）
    logp: torch.Tensor        # (B, T) 旧策略 log π
    values: torch.Tensor      # (B, T)
    rewards: torch.Tensor     # (B, T)
    adv: torch.Tensor         # (B, T)
    returns: torch.Tensor     # (B, T)
    rm_score: torch.Tensor    # (B,)

@torch.no_grad()
def sample_responses(policy, prompts, cfg, generator):
    """从策略采样完整响应，返回 ``prompt + response`` 的 Token 序列。

    采样分布就是模型在全词表上的 softmax，不做词表截断，也不强制 EOS；
    这样记录的 log 概率与实际采样分布严格一致。到达 ``max_new`` 仍没有
    EOS 的序列按有限期回合截断，最后一个 Token 视为终止状态。
    """
    policy.eval()
    seqs = prompts
    done = torch.zeros(seqs.size(0), dtype=torch.bool)
    for _ in range(cfg.max_new):
        logits = policy(seqs).logits[:, -1]
        probs = torch.softmax(logits / cfg.temperature, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, cfg.pad_id), nxt)
        seqs = torch.cat([seqs, nxt[:, None]], dim=1)
        done = done | (nxt == cfg.eos_id)
        if bool(done.all()):
            break
    return seqs

def response_mask(seqs, prompt_len, pad_id):
    """响应位置：位于提示之后且不是 padding；EOS 计入响应。"""
    positions = torch.arange(seqs.size(1), device=seqs.device)[None, :]
    return (seqs != pad_id) & (positions >= prompt_len)

def token_logprobs(model, seqs, attn):
    """返回每个位置预测*下一个* Token 的 log 概率，形状 ``(B, T)``。

    第 0 位没有前文，填 0；其余位置满足 logp[:, t] = log π(seqs[t] | seqs[:t])。
    padding 位置填 0，避免 ``-inf`` 与掩码相乘产生 NaN。
    """
    logits = model(seqs, attention_mask=attn).logits[:, :-1]
    logp = F.log_softmax(logits, dim=-1)
    target = seqs[:, 1:]
    out = logp.gather(-1, target[..., None]).squeeze(-1)
    out = torch.where(attn[:, 1:], out, torch.zeros_like(out))
    return F.pad(out, (1, 0))

def token_values(value, seqs, attn):
    """把位置 t 的价值对齐到「预测 Token t」：V(s_{t-1})。"""
    values = value(seqs, attention_mask=attn)
    return F.pad(values[:, :-1], (1, 0))

def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """(16.8)：广义优势估计。

    $$\\delta_t=r_t+\\gamma V(s_{t+1})-V(s_t)$$，
    $$A_t=\\sum_l (\\gamma\\lambda)^l\\delta_{t+l}$$。响应结束后的
    padding 位置把优势与自举价值都置零。
    """
    batch, length = rewards.shape
    adv = torch.zeros_like(rewards)
    running = torch.zeros(batch, device=rewards.device)
    for t in reversed(range(length)):
        if t + 1 < length:
            next_value = torch.where(mask[:, t + 1], values[:, t + 1],
                                     torch.zeros_like(values[:, t + 1]))
        else:
            next_value = torch.zeros(batch, device=rewards.device)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        running = (delta + gamma * lam * running) * mask[:, t].to(running.dtype)
        adv[:, t] = running
    returns = adv + values
    return adv, returns

def normalize_advantage(adv, mask):
    """按批内有效位置做零均值、单位标准差；padding 位置保持 0。"""
    x = adv[mask]
    if x.numel() == 0:
        return torch.zeros_like(adv)
    norm = (adv - x.mean()) / (x.std(unbiased=False) + 1e-8)
    return norm * mask.to(norm.dtype)

def clipped_surrogate(ratio, adv, clip):
    """(16.7)：$$\\min(\\rho A,\\ \\mathrm{clip}(\\rho,1-\\epsilon,1+\\epsilon)A)$$。"""
    # (16.7)
    return torch.minimum(ratio * adv,
                         torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv)

def kl_penalty(logp, ref_logp, mask):
    """(16.10)：k1 估计量 $$\\log\\pi-\\log\\pi_{\\mathrm{ref}}$$ 的批平均。

    在旧策略的样本上计算，不需要额外采样；逐 Token 的 log 比同时用作
    (16.5) 的塑形奖励。k1 无偏但方差大，单个 Token 可以为负，只有批
    平均才逼近 $$\\mathrm{KL}(\\pi\\|\\pi_{\\mathrm{ref}})$$。
    """
    # (16.10)
    return ((logp - ref_logp) * mask).sum() / mask.sum().clamp_min(1)

def kl_estimator_k3(logp, ref_logp, mask):
    """(16.11)：k3 估计量 $$r-\\log r-1$$，其中 $$r=\\pi_{\\mathrm{ref}}/\\pi$$。

    逐 Token 非负，方差显著小于 k1；代价是需要对 log 比取指数。
    """
    # (16.11)
    log_ratio = torch.clamp(ref_logp - logp, -20.0, 20.0)
    k3 = torch.exp(log_ratio) - log_ratio - 1.0
    return ((k3 * mask).sum() / mask.sum().clamp_min(1),
            (k3 * mask).sum(1) / mask.sum(1).clamp_min(1))

def collect_rollout(policy, ref, value, reward_model, prompts, cfg, generator):
    """采样一批响应并计算 log 概率、价值、KL 塑形奖励与 GAE 优势。"""
    prompt_len = prompts.size(1)
    seqs = sample_responses(policy, prompts, cfg, generator)
    attn = seqs != cfg.pad_id
    mask = response_mask(seqs, prompt_len, cfg.pad_id)

    with torch.no_grad():
        logp = token_logprobs(policy, seqs, attn)
        ref_logp = token_logprobs(ref, seqs, attn)
        values = token_values(value, seqs, attn)
        rm_score = reward_model.score(seqs, attn)

    if cfg.normalize_reward:
        rm_score = (rm_score - rm_score.mean()) / (
            rm_score.std(unbiased=False) + 1e-6)

    # (16.5)：逐 Token 奖励 = -β KL_t，序列末尾加奖励模型的分数
    rewards = -cfg.beta * (logp - ref_logp) * mask
    last = attn.long().sum(-1) - 1
    rewards.scatter_add_(1, last[:, None], rm_score[:, None])
    values = values * mask

    adv, returns = compute_gae(rewards, values, mask, cfg.gamma, cfg.lam)
    return Rollout(seqs=seqs, attn=attn, mask=mask, logp=logp.detach(),
                   values=values, rewards=rewards, adv=adv,
                   returns=returns, rm_score=rm_score)

def ppo_update(policy, value, opt_pi, opt_v, batch, cfg, generator):
    """(16.6)–(16.9)：在固定 rollout 上做若干轮 PPO 更新，返回诊断量。"""
    batch_size = batch.seqs.size(0)
    stats = []
    for _ in range(cfg.epochs):
        order = torch.randperm(batch_size, generator=generator)
        for start in range(0, batch_size, cfg.minibatch):
            idx = order[start:start + cfg.minibatch]
            seqs, attn, mask = batch.seqs[idx], batch.attn[idx], batch.mask[idx]
            logp = token_logprobs(policy, seqs, attn)
            values = token_values(value, seqs, attn)

            # (16.6)
            ratio = torch.exp(logp - batch.logp[idx])
            adv = normalize_advantage(batch.adv[idx], mask)
            # (16.9)：策略损失 + 价值回归
            pi_loss = -(clipped_surrogate(ratio, adv, cfg.clip) * mask).sum() \
                / mask.sum().clamp_min(1)
            v_loss = ((values - batch.returns[idx]) ** 2 * mask).sum() \
                / mask.sum().clamp_min(1)
            loss = pi_loss + cfg.vf_coef * v_loss

            opt_pi.zero_grad(set_to_none=True)
            opt_v.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(value.parameters(), cfg.grad_clip)
            opt_pi.step()
            opt_v.step()
            stats.append((float(pi_loss.detach()), float(v_loss.detach()),
                          float(ratio[mask].mean().detach())))
    return stats
