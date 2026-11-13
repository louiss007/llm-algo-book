"""第 16 章：奖励模型与 Bradley-Terry 损失（tinylm/posttrain/rm.py）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model import TinyLM

class ScalarModel(nn.Module):
    """把 TinyLM 的词表头换成标量头。

    奖励模型与价值模型都与策略共享同一套 Transformer 主干，只是输出维
    不同：策略输出 ``(B, T, n_vocab)``，这里输出 ``(B, T)``。替换
    ``lm_head`` 而不改 ``TinyLM.forward``，因此三份模型可以复用同一套
    位置编码、注意力与因果掩码。
    """

    def __init__(self, cfg):
        super().__init__()
        assert not cfg.tie_embeddings, "标量头无法与词表嵌入绑定"
        self.cfg = cfg
        self.backbone = TinyLM(cfg)
        self.backbone.lm_head = nn.Linear(cfg.d_model, 1, bias=False)
        nn.init.normal_(self.backbone.lm_head.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        return self.backbone(
            input_ids, attention_mask=attention_mask).logits.squeeze(-1)

    def score(self, input_ids, attention_mask):
        """取每条序列最后一个有效位置（EOS）的标量输出。"""
        values = self(input_ids, attention_mask=attention_mask)
        last = attention_mask.long().sum(-1) - 1
        return values.gather(1, last[:, None]).squeeze(1)

def init_scalar_from_lm(cfg, lm_state, seed=0):
    """用 SFT 策略初始化奖励/价值主干，标量头保持随机。

    ``lm_head`` 的形状从 ``(n_vocab, d_model)`` 变成 ``(1, d_model)``，
    不能直接加载；其余参数逐位复制。返回被丢弃的键，便于测试。
    """
    torch.manual_seed(seed)
    model = ScalarModel(cfg)
    dropped = [k for k in lm_state if k.startswith("lm_head.")]
    keep = {k: v for k, v in lm_state.items() if not k.startswith("lm_head.")}
    missing, unexpected = model.backbone.load_state_dict(keep, strict=False)
    assert not unexpected, unexpected
    # 主干之外只有被替换的 lm_head 应当缺失
    assert set(missing) <= {"lm_head.weight"}, missing
    nn.init.normal_(model.backbone.lm_head.weight, std=0.02)
    return model, dropped

def bradley_terry_loss(reward_win, reward_lose):
    """(16.1)：$$-\\log\\sigma(r_w-r_l)$$ 的批平均。

    只依赖奖励差，因此对全体奖励加常数不变；这是奖励尺度不可识别的
    来源，也是 DPO 能从同一目标出发消掉奖励的原因。
    """
    # (16.1)
    return -F.logsigmoid(reward_win - reward_lose).mean()

def pairwise_accuracy(reward_win, reward_lose):
    """偏好对上的排序准确率；平局按失败计。"""
    return (reward_win > reward_lose).float().mean()

def preference_probability(reward_win, reward_lose):
    """(16.1) 给出的 $$P(y_w\\succ y_l)$$。"""
    return torch.sigmoid(reward_win - reward_lose)

def reward_standardizer(rewards):
    """返回 $$(r-\\mu)/\\sigma$$ 所需的两个标量。

    奖励的零点与尺度都是任意的；PPO 只用奖励差与相对大小，标准化
    不改变最优策略，但能让价值网络的回归目标落在稳定区间。
    """
    mean = rewards.mean()
    std = rewards.std(unbiased=False).clamp_min(1e-6)
    return mean, std
