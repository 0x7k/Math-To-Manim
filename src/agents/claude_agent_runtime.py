"""Utility helpers for interacting with Claude via the Agent SDK.

The Math-To-Manim agents historically spoke to the Anthropic Messages API
directly. Certain newer Claude releases are distributed primarily through the
Claude Code / Agent SDK interface, which does not require specifying an exact
`model=` name.  This module provides a minimal wrapper so existing synchronous
call sites can fall back to the Agent SDK whenever the legacy endpoint is
unavailable (for example, when a Messages API request returns a 404 for a
model that now lives exclusively in Claude Code).

用于通过 Agent SDK 与 Claude 交互的实用工具函数。
Math-To-Manim 代理程序以前是直接与 Anthropic Messages API 对话的。
某些较新的 Claude 版本主要通过 Claude Code / Agent SDK 接口发布，这个接口不需要指定确切的 model= 名称。
本模块提供了一个极简的封装器，因此现有的同步调用站点可以在传统端点不可用时
（例如，当 Messages API 请求对一个现已仅存在于 Claude Code 中的模型返回 404 错误时）回退到使用 Agent SDK。
"""

from __future__ import annotations

import asyncio
from typing import Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    ToolResultBlock,
    query,
)


async def _run_query_async(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Internal async helper that streams responses from Claude Code.
    内部异步辅助函数，用于从 Claude Code 流式传输响应
    """

    options = ClaudeAgentOptions()
    if system_prompt is not None:
        options.system_prompt = system_prompt

    # Pass temperature/max tokens as extra CLI arguments when provided.
    if temperature is not None:
        options.extra_args["temperature"] = str(temperature)
    if max_tokens is not None:
        options.extra_args["max-tokens"] = str(max_tokens)

    chunks: list[str] = []

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
                elif isinstance(block, ToolResultBlock):
                    content = block.content
                    if isinstance(content, str):
                        chunks.append(content)
                    elif isinstance(content, list):
                        for item in content:
                            text = item.get("text")
                            if text:
                                chunks.append(text)

    return "".join(chunks).strip()


def run_query_via_sdk(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Public synchronous wrapper used by legacy call sites.
    旧版调用点使用的公共同步包装器
    """

    return asyncio.run(
        _run_query_async(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )


__all__ = ["run_query_via_sdk"]


