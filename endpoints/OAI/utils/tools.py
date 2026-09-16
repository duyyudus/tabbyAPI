"""Tool call processing utilities for OAI server."""

import json
import re

from common.logger import xlogger
from typing import List

from endpoints.OAI.types.tools import ToolCall
from endpoints.OAI.utils.toolcall_formats import (
    qwen3_coder,
    minimax_m2,
    glm4_5,
    harmony,
    hy3,
    muse_glimmer,
    deepseek_v4,
    mistral_old,
    mistral,
    gemma4,
    lfm2,
)

ALL_TOOLCALL_FORMATS = {
    "deepseek_v4": deepseek_v4,
    "dsv4": deepseek_v4,
    "gemma4": gemma4,
    "glm4_5": glm4_5,
    "glm4_6": glm4_5,
    "glm4_7": glm4_5,
    "harmony": harmony,
    "hy3": hy3,
    "hy_v3": hy3,
    "laguna": glm4_5,
    "poolside_v1": glm4_5,
    "lfm": lfm2,
    "lfm2": lfm2,
    "lfm2_5": lfm2,
    "minimax_m2": minimax_m2,
    "minimax_m2_1": minimax_m2,
    "minimax_m2_5": minimax_m2,
    "mistral_old": mistral_old,
    "mistral": mistral,
    "muse_glimmer": muse_glimmer,
    "glimmer": muse_glimmer,
    "qwen3_coder": qwen3_coder,
    "qwen3_5": qwen3_coder,
    "step3_5": qwen3_coder,
    "step3_7": qwen3_coder,
}


def _get_parser(tool_format: str):
    if not tool_format:
        return None
    parser = ALL_TOOLCALL_FORMATS.get(tool_format)
    if not parser:
        xlogger.error(f"Unknown tool format given: {tool_format}")
    return parser


def get_toolcall_tags(tool_format: str):
    parser = _get_parser(tool_format)
    if not parser:
        return None, None
    return parser.TOOLCALL_START, parser.TOOLCALL_END


def is_supported_format(tool_format: str) -> bool:
    return tool_format in ALL_TOOLCALL_FORMATS


def parse_toolcalls(tool_calls_str: str, tool_format: str, strict: bool = False) -> List[ToolCall]:
    """
    Dispatch tool call parsing to the appropriate format handler.

    Args:
        tool_calls_str: Raw tool call text from model generation.
        tool_format: See below

    Returns:
        List of parsed ToolCall objects. Empty list on parse failure (never raises).
    """

    try:
        parser = _get_parser(tool_format)
        if not parser:
            if strict:
                raise ValueError("No tool parser configured")
            return []

        calls = parser.parse_toolcalls(tool_calls_str)
        if strict:
            _validate_complete_calls(tool_calls_str, parser, calls)
        return calls

    except Exception as e:
        if strict:
            raise
        xlogger.error(
            "ToolCallProcessor.parse: Failed to parse tool calls",
            {"tool_format": tool_format, "e": str(e)},
            details=f"(format={tool_format}): {e}",
        )
        return []


def _validate_complete_calls(text, parser, calls):
    """Reject partial parses; legacy chat parsers intentionally tolerate malformed calls."""
    if not calls:
        raise ValueError("No complete tool calls could be parsed")
    start, end = parser.TOOLCALL_START, parser.TOOLCALL_END
    if start and end and text.count(start) != text.count(end):
        raise ValueError("Unclosed tool call wrapper")
    markers = {
        qwen3_coder: ("<function=", "</function>"),
        minimax_m2: ("<invoke ", "</invoke>"),
        deepseek_v4: ("<｜DSML｜invoke ", "</｜DSML｜invoke>"),
        muse_glimmer: ("<atem:invoke ", "</atem:invoke>"),
        hy3: ("<tool_call:opensource>", "</tool_call:opensource>"),
        glm4_5: ("<tool_call>", "</tool_call>"),
        gemma4: ("<|tool_call>", "<tool_call|>"),
        harmony: ("<|message|>", "<|call|>"),
    }
    if parser in markers:
        opening, closing = markers[parser]
        if text.count(opening) != len(calls) or text.count(closing) != len(calls):
            raise ValueError("Tool parser could not consume every call")
    if parser is qwen3_coder and text.count(start) != len(calls):
        raise ValueError("Malformed Qwen tool call wrapper")
    if parser is mistral and text.count(start) != len(calls):
        raise ValueError("Tool parser could not consume every Mistral call")
    if parser is mistral_old:
        expected = json.loads(text.removeprefix(start).strip())
        if not isinstance(expected, list) or len(expected) != len(calls):
            raise ValueError("Tool parser could not consume every Mistral call")
    # Missing/duplicate parameter tags must not silently turn into fewer arguments.
    parameter_tags = {
        qwen3_coder: (r"<parameter=[^>]+>", "</parameter>"),
        minimax_m2: (r"<parameter [^>]+>", "</parameter>"),
        deepseek_v4: (r"<｜DSML｜parameter [^>]+>", "</｜DSML｜parameter>"),
        muse_glimmer: (r"<atem:parameter [^>]+>", "</atem:parameter>"),
        glm4_5: (r"<arg_key>", "</arg_key>"),
        hy3: (r"<arg_key:opensource>", "</arg_key:opensource>"),
    }
    if parser in parameter_tags:
        opening, closing = parameter_tags[parser]
        count = len(re.findall(opening, text))
        if count != text.count(closing) or count != sum(
            len(json.loads(call.function.arguments)) for call in calls
        ):
            raise ValueError("Malformed or duplicate tool parameters")
        if parser in (glm4_5, hy3):
            suffix = ":opensource" if parser is hy3 else ""
            if count != text.count(f"<arg_value{suffix}>") or count != text.count(
                f"</arg_value{suffix}>"
            ):
                raise ValueError("Unmatched tool argument values")
