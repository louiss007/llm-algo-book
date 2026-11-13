"""分词器协议（tinylm/tokenizer/base.py，附录 F.2.1）。"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    vocab_size: int

    def encode(self, text: str, *, add_special: bool = True) -> list[int]: ...
    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str: ...
    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "Tokenizer": ...
