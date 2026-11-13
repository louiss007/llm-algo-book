"""第 14 章：对话模板与损失掩码（tinylm/tokenizer/chat.py）。"""
from dataclasses import dataclass

import torch

IGNORE_INDEX = -100


@dataclass(frozen=True)
class ChatTemplate:
    """把多轮对话渲染成一条 Token 序列。

    ``turns`` 是 ``(role, content)`` 的列表，``role`` 取 ``"user"`` 或
    ``"assistant"``，``content`` 是 Token ID 列表。模板只负责拼接与
    掩码，不负责把文本切成 Token；分词由第 2 章的 ``Tokenizer`` 完成。
    """

    name: str = "default"
    bos_token: int = 1
    user_token: int = 2
    assistant_token: int = 3
    eos_token: int = 4
    system_token: int | None = None

    def render(self, turns, *, answer_only=True):
        """返回 ``input_ids`` 与 ``labels``。

        只有 assistant 回合的内容与回合结束符进入损失；user 回合、
        role 标记、BOS 以及 padding 位置在 ``labels`` 中都是 ``-100``。
        标签由输入右移一位的目标构造；最后一个位置没有目标，固定为
        ``-100``。
        """
        ids, mask = [self.bos_token], [0]
        for role, content in turns:
            if role == "system":
                if self.system_token is None:
                    raise ValueError("模板没有 system_token")
                ids.append(self.system_token)
                mask.append(0)
            elif role == "user":
                ids.append(self.user_token)
                mask.append(0)
            elif role == "assistant":
                ids.append(self.assistant_token)
                mask.append(0)
            else:
                raise ValueError(f"未知角色 {role!r}")
            ids.extend(int(t) for t in content)
            content_supervised = (role == "assistant") or (not answer_only)
            mask.extend([1 if content_supervised else 0] * len(content))
            if role == "assistant":
                ids.append(self.eos_token)
                mask.append(1)

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        next_supervised = torch.tensor(mask[1:], dtype=torch.bool)
        labels[:-1] = torch.where(
            next_supervised, input_ids[1:],
            torch.full_like(input_ids[1:], IGNORE_INDEX))
        return {"input_ids": input_ids, "labels": labels}


def encode_conversation(turns, template=None, *, answer_only=True):
    """模板渲染的简写；``template=None`` 时使用默认模板。"""
    return (template or ChatTemplate()).render(
        turns, answer_only=answer_only)


def count_supervised(rendered):
    """(14.3)：返回 (监督标签数, 非 padding 输入数, 监督比例)。"""
    n_sup = int((rendered["labels"] != IGNORE_INDEX).sum())
    n_tok = int(rendered["input_ids"].numel())
    return n_sup, n_tok, (n_sup / n_tok if n_tok else 0.0)
