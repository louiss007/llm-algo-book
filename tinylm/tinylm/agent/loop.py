"""第 26 章：最小智能体与工具调用循环（tinylm/agent/loop.py）。

只依赖 Python 标准库。策略、工具与循环彼此解耦：
``run_agent`` 接收任意 ``policy(context) -> str``，不绑定具体模型。
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

class ActionParseError(ValueError):
    """策略输出无法解析为工具调用或最终答案。"""

@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 6
    max_context_tokens: int = 4096
    max_repeat: int = 2
    max_observation_chars: int = 512
    stop_on_tool_error: bool = False

    def __post_init__(self):
        assert self.max_steps >= 1
        assert self.max_context_tokens >= 1
        assert self.max_repeat >= 1
        assert self.max_observation_chars >= 1

@dataclass(frozen=True)
class ToolSpec:
    """(26.1)：工具是名字、参数模式与可调用函数的组合。"""

    name: str
    description: str
    parameters: Mapping[str, type]
    func: Callable[..., Any]

    def __post_init__(self):
        assert self.name.isidentifier()
        assert callable(self.func)
        assert all(isinstance(k, str) and v in (str, int, float, bool)
                   for k, v in self.parameters.items())

@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]

@dataclass(frozen=True)
class FinalAction:
    answer: str

@dataclass(frozen=True)
class AgentStep:
    index: int
    action: str
    observation: str | None
    context_tokens: int
    action_tokens: int
    error: bool

@dataclass(frozen=True)
class AgentResult:
    success: bool
    answer: str | None
    steps: tuple[AgentStep, ...]
    stop_reason: str

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def context_tokens(self) -> int:
        return sum(step.context_tokens for step in self.steps)

def _error(message: str) -> str:
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)

def _is_error(observation: str) -> bool:
    return observation.startswith('{"ok": false')

def _truncate(payload: dict[str, Any], max_chars: int) -> str:
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return json.dumps({"ok": payload.get("ok", False), "truncated": True,
                       "preview": text[:max_chars]}, ensure_ascii=False)

def validate_arguments(tool: ToolSpec, args: Mapping[str, Any]) -> dict[str, Any]:
    expected = dict(tool.parameters)
    missing = sorted(set(expected) - set(args))
    unknown = sorted(set(args) - set(expected))
    if missing or unknown:
        raise ValueError(f"missing={missing}, unknown={unknown}")
    for name, kind in expected.items():
        value = args[name]
        if kind is bool:
            ok = isinstance(value, bool)
        elif kind is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif kind is float:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            ok = isinstance(value, str)
        if not ok:
            raise TypeError(f"argument {name!r} must be {kind.__name__}")
    return dict(args)

class ToolRegistry:
    """按名字注册与执行工具；参数校验失败时不调用函数。"""

    def __init__(self, tools: Sequence[ToolSpec] = ()):
        self._tools: dict[str, ToolSpec] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def describe(self) -> str:
        return "\n".join(
            f"- {tool.name}({', '.join(f'{k}: {v.__name__}' for k, v in tool.parameters.items())}): "
            f"{tool.description}"
            for tool in sorted(self._tools.values(), key=lambda t: t.name))

    def execute(self, call: ToolCall, max_chars: int = 512) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            return _error(f"unknown tool: {call.name}")
        try:
            args = validate_arguments(tool, call.args)
        except (ValueError, TypeError) as exc:
            return _error(f"invalid arguments: {exc}")
        try:
            result = tool.func(**args)
        except Exception as exc:  # 工具失败也是一种观测，不能让循环崩掉
            return _error(f"{type(exc).__name__}: {exc}")
        return _truncate({"ok": True, "result": result}, max_chars)

def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ActionParseError("no JSON object found")

def parse_action(text: str) -> ToolCall | FinalAction:
    """接受“先写推理、再写一个 JSON 对象”的输出格式。"""
    obj = extract_json_object(text)
    has_answer = "answer" in obj
    has_tool = "tool" in obj
    if has_answer == has_tool:
        raise ActionParseError("exactly one of 'answer' or 'tool' is required")
    if has_answer:
        return FinalAction(str(obj["answer"]))
    name = obj["tool"]
    args = obj.get("args", {})
    if not isinstance(name, str) or not isinstance(args, dict):
        raise ActionParseError("'tool' must be a string and 'args' an object")
    return ToolCall(name, args)

def estimate_tokens(text: str) -> int:
    """按 UTF-8 字节数除以 4 估算 Token 数；只用于账本，不替代分词器。"""
    return max(1, (len(text.encode("utf-8")) + 3) // 4)

def run_agent(task: str, policy: Callable[[str], str],
              registry: ToolRegistry, cfg: AgentConfig = AgentConfig(),
              token_counter: Callable[[str], int] = estimate_tokens) -> AgentResult:
    """执行“策略—工具—观测”循环，直到给出答案或触发停止条件。"""
    context = f"task: {task}"                      # (26.2)
    steps: list[AgentStep] = []
    seen: Counter[tuple[str, str]] = Counter()

    for index in range(cfg.max_steps):
        context_tokens = token_counter(context)
        if context_tokens > cfg.max_context_tokens:
            return AgentResult(False, None, tuple(steps), "context_limit")

        raw = policy(context)
        action_tokens = token_counter(raw)

        try:
            action = parse_action(raw)
        except ActionParseError as exc:
            observation = _error(f"parse: {exc}")
            context += f"\naction: {raw}\nobservation: {observation}"
            steps.append(AgentStep(index, raw, observation, context_tokens,
                                   action_tokens, True))
            if cfg.stop_on_tool_error:
                return AgentResult(False, None, tuple(steps), "tool_error")
            continue

        if isinstance(action, FinalAction):
            steps.append(AgentStep(index, raw, None, context_tokens,
                                   action_tokens, False))
            return AgentResult(True, action.answer, tuple(steps), "final")

        key = (action.name, json.dumps(action.args, sort_keys=True,
                                       ensure_ascii=False))
        seen[key] += 1
        if seen[key] > cfg.max_repeat:
            observation = _error(f"repeated call rejected: {action.name}")
            error = True
        else:
            observation = registry.execute(action, cfg.max_observation_chars)
            error = _is_error(observation)

        steps.append(AgentStep(index, raw, observation, context_tokens,
                               action_tokens, error))
        context += f"\naction: {raw}\nobservation: {observation}"   # (26.5)

        if error and cfg.stop_on_tool_error:
            return AgentResult(False, None, tuple(steps), "tool_error")

    return AgentResult(False, None, tuple(steps), "max_steps")

def trajectory_cost(context_tokens: Sequence[int],
                    action_tokens: Sequence[int],
                    c_in: float = 1.0, c_out: float = 4.0) -> float:
    """(26.7)：每一步都要重发历史，输入成本随步数累积。"""
    assert len(context_tokens) == len(action_tokens)
    assert all(x >= 0 for x in context_tokens)
    assert all(x >= 0 for x in action_tokens)
    assert c_in >= 0 and c_out >= 0
    return sum(c_in * length + c_out * action
               for length, action in zip(context_tokens, action_tokens))

def result_cost(result: AgentResult, c_in: float = 1.0,
                c_out: float = 4.0) -> float:
    return trajectory_cost([step.context_tokens for step in result.steps],
                           [step.action_tokens for step in result.steps],
                           c_in, c_out)

def retry_success(p: float, attempts: int) -> float:
    """(26.8)：一次尝试成功率 p，最多 attempts 次时的成功率。"""
    assert 0.0 <= p <= 1.0
    assert attempts >= 1
    return 1.0 - (1.0 - p) ** attempts

def task_success(p: float, attempts: int, steps: int) -> float:
    """(26.9)：K 个独立步骤全部成功的概率。"""
    assert steps >= 1
    return retry_success(p, attempts) ** steps

def value_of_information(utility_before: float, utility_after: float,
                         cost: float) -> float:
    """(26.10)：调用工具带来的单位成本期望收益。"""
    assert cost > 0
    return (utility_after - utility_before) / cost
