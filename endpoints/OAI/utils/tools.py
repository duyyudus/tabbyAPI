"""Tool call processing utilities for OAI server."""

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from common.logger import xlogger

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
    hermes,
    kimi,
    olmo3,
)

ALL_TOOLCALL_FORMATS = {
    "deepseek_v4": deepseek_v4,
    "dsv4": deepseek_v4,
    "gemma4": gemma4,
    "glm4_5": glm4_5,
    "glm4_6": glm4_5,
    "glm4_7": glm4_5,
    "harmony": harmony,
    "hermes": hermes,
    "qwen3": hermes,
    "qwen2_5": hermes,
    "smollm3": hermes,
    "kimi": kimi,
    "kimi_k2": kimi,
    "kimi_linear": kimi,
    "olmo3": olmo3,
    "olmo": olmo3,
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
    "spark2_5": glm4_5,
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


def canonical_format_name(tool_format: str) -> str:
    """The registry's primary name for a format or alias."""

    parser = ALL_TOOLCALL_FORMATS.get(tool_format)
    if parser is None:
        return tool_format
    return parser.__name__.rsplit(".", 1)[-1]


def format_reasoning_tags(tool_format: Optional[str]) -> Optional[tuple]:
    """Reasoning tags the models using this format are trained with, if known."""

    parser = _get_parser(tool_format) if tool_format else None
    signature = getattr(parser, "DETECT", None)
    return signature.reasoning_tags if signature else None


# Reasoning tag pairs known to TabbyAPI, scanned when no tool format supplies
# them. Order matters only for the unlikely case of a template containing two
KNOWN_REASONING_TAGS = (
    ("<think>", "</think>"),
    ("[THINK]", "[/THINK]"),
    ("<|channel>thought", "<channel|>"),
    ("<think:opensource>", "</think:opensource>"),
    ("<reasoning>", "</reasoning>"),
    ("<|thinking|>", "</thinking>"),
)


@dataclass
class FormatDetection:
    """Outcome of tool format auto-detection."""

    tool_format: Optional[str] = None
    score: int = 0
    evidence: List[str] = field(default_factory=list)
    # Other formats that scored the same, when the result was ambiguous
    ambiguous: List[str] = field(default_factory=list)


# Evidence weights. A template match is decisive on its own; tokens alone can
# select a format (special-token formats without a template); the architecture
# only breaks ties between the two
_TEMPLATE_WEIGHT = 4
_TOKEN_WEIGHT = 2
_ARCH_WEIGHT = 1


def _score_format(
    signature, template: str, has_token: Callable[[str], bool], architecture: str
) -> tuple[int, List[str]]:
    score, evidence = 0, []

    def present(marker) -> bool:
        alternatives = marker if isinstance(marker, tuple) else (marker,)
        return any(alt in template for alt in alternatives)

    if signature.template_markers and template:
        if all(present(m) for m in signature.template_markers) and not any(
            m in template for m in signature.template_exclude
        ):
            score += _TEMPLATE_WEIGHT
            evidence.append("template")

    if signature.special_tokens and all(has_token(t) for t in signature.special_tokens):
        score += _TOKEN_WEIGHT
        evidence.append("tokenizer")

    if architecture and any(a.lower() in architecture.lower() for a in signature.architectures):
        score += _ARCH_WEIGHT
        evidence.append("architecture")

    return score, evidence


def detect_tool_format(
    template: Optional[str],
    has_token: Callable[[str], bool],
    architecture: Optional[str] = None,
) -> FormatDetection:
    """
    Pick the tool call format for a model from its chat template, tokenizer and
    architecture. Every registered format is scored on which of its signature's
    evidence is present; the best score wins if it is unique and rests on more
    than the architecture name alone.
    """

    # A template that defines tools is the authority on how calls are written:
    # tokens and architecture alone must not pick a format it doesn't render
    template_defines_tools = bool(template) and "tools" in template

    scored = []
    seen = set()
    for name, module in ALL_TOOLCALL_FORMATS.items():
        if id(module) in seen:
            continue
        seen.add(id(module))
        signature = getattr(module, "DETECT", None)
        if signature is None:
            continue
        score, evidence = _score_format(signature, template or "", has_token, architecture or "")
        if score <= _ARCH_WEIGHT:
            continue
        if template_defines_tools and "template" not in evidence:
            continue
        scored.append((score, canonical_format_name(name), evidence))

    if not scored:
        return FormatDetection()

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_name, best_evidence = scored[0]
    ties = [name for score, name, _ in scored[1:] if score == best_score]
    if ties:
        return FormatDetection(score=best_score, ambiguous=[best_name, *ties])

    return FormatDetection(tool_format=best_name, score=best_score, evidence=best_evidence)


def detect_reasoning_tags(
    template: Optional[str],
    has_token: Callable[[str], bool],
    tool_format: Optional[str] = None,
) -> tuple[Optional[tuple], Optional[str]]:
    """
    Find the reasoning tags a model uses. The tool format's known tags win when
    the template or tokenizer confirms them; otherwise every known pair is
    tried the same way. Returns (tags, evidence) or (None, None).
    """

    template = template or ""

    def confirmed(tags):
        start, end = tags
        if start in template and end in template:
            return "template"
        if has_token(start) and has_token(end):
            return "tokenizer"
        return None

    format_tags = format_reasoning_tags(tool_format)
    candidates = ([format_tags] if format_tags else []) + list(KNOWN_REASONING_TAGS)
    for tags in candidates:
        evidence = confirmed(tags)
        if evidence:
            return tags, evidence

    return None, None


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
        hermes: ("<tool_call>", "</tool_call>"),
        kimi: ("<|tool_call_begin|>", "<|tool_call_end|>"),
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
    # JSON argument parsers wrap undecodable arguments as {"input": raw}; reject those.
    if parser is hermes:
        for block in hermes._OUTER.findall(text):
            call = hermes._parse_block(block)
            _require_json_object(call.get("arguments", call.get("parameters", {})), allow_null=True)
    if parser is kimi:
        for match in kimi._CALL.finditer(text):
            _require_json_object(match.group("args") or "{}")
    if parser is olmo3:
        _validate_olmo3_calls(text, calls)
    if parser is lfm2:
        _validate_lfm2_calls(text, calls)
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


def _require_json_object(args, allow_null=False):
    """Reject tool arguments that are not a JSON object (or JSON text of one)."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            raise ValueError("Tool call arguments are not valid JSON") from None
    if args is None and allow_null:
        return
    if not isinstance(args, dict):
        raise ValueError("Tool call arguments are not a JSON object")


def _olmo3_call_lines(text):
    """Split each <function_calls> block into calls, joining lines as the olmo3 parser does."""
    for block in olmo3._OUTER.findall(text):
        pending = ""
        for line in block.splitlines():
            pending = f"{pending}\n{line}" if pending else line
            if pending.count("(") <= pending.count(")"):
                if pending.strip():
                    yield pending
                pending = ""
        if pending.strip():
            yield pending


def _validate_olmo3_calls(text, calls):
    """The olmo3 parser drops unparseable lines and arguments without a key."""
    lines = list(_olmo3_call_lines(text))
    if len(lines) != len(calls):
        raise ValueError("Tool parser could not consume every call")
    for line in lines:
        match = olmo3._CALL.match(line.strip())
        keys = []
        for part in olmo3._split_args(match.group(2)):
            key = olmo3._KEY.match(part)
            if not key:
                raise ValueError("Malformed tool parameters")
            keys.append(key.group(1))
        if len(keys) != len(set(keys)):
            raise ValueError("Malformed or duplicate tool parameters")


def _validate_lfm2_calls(text, calls):
    """The lfm2 parser skips unparseable lists and ignores positional or non-literal values."""
    total = 0
    for block in lfm2._extract_tool_texts(text):
        module, _ = lfm2._safe_parse_list(block)
        body = module.body if module else []
        node = getattr(body[0], "value", None) if len(body) == 1 else None
        if not (isinstance(node, ast.List) and all(isinstance(e, ast.Call) for e in node.elts)):
            raise ValueError("Malformed tool call list")
        for call in node.elts:
            keys = [kw.arg for kw in call.keywords]
            if call.args or None in keys:
                raise ValueError("Tool call arguments must be keyword arguments")
            if len(keys) != len(set(keys)):
                raise ValueError("Malformed or duplicate tool parameters")
            for kw in call.keywords:
                try:
                    ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    raise ValueError("Tool call argument is not a literal value") from None
        total += len(node.elts)
    if total != len(calls):
        raise ValueError("Tool parser could not consume every call")
