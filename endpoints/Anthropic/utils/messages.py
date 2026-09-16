"""Messages event accumulation and serialization over the shared generation iterator."""

import asyncio
import json
from uuid import uuid4

import anyio

from common.logger import xlogger
from endpoints.OAI.utils.chat_completion import _resolve_start_in_reasoning, iter_generation_events
from endpoints.OAI.utils.responses_input import InvalidModelOutput
from endpoints.OAI.utils.stream_parser import CONTENT, REASONING, TOOL
from endpoints.OAI.utils.tools import parse_toolcalls


class MessageGenerationError(RuntimeError):
    """Generation failed after validation; reported as api_error."""


def usage(input_tokens, output_tokens=0, cached_tokens=0):
    # Anthropic reports cache reads separately from input_tokens.
    return {
        "input_tokens": input_tokens - cached_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
    }


class MessagesAccumulator:
    def __init__(self, data, tools, model_name, input_tokens):
        self.tools = tools
        self.stop_sequences = getattr(data, "stop_sequences", None) or []
        thinking = data.thinking
        self.show_thinking = thinking is not None and thinking.type != "disabled"
        self.omit_thinking = self.show_thinking and thinking.display == "omitted"
        self.message = {
            "id": "msg_" + uuid4().hex[:24],
            "type": "message",
            "role": "assistant",
            "model": model_name,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": usage(input_tokens),
        }
        self.index = -1
        self.open = None
        self.held_whitespace = ""
        self.deferred = False
        self.segments = []
        self.tool_format = None
        self.finished = False

    def start(self):
        message = {**self.message, "usage": dict(self.message["usage"])}
        return [{"type": "message_start", "message": message}]

    def open_block(self, block):
        events = self.close_block()
        self.index += 1
        self.open = block["type"]
        events.append({"type": "content_block_start", "index": self.index, "content_block": block})
        return events

    def close_block(self):
        if self.open is None:
            return []
        events = []
        if self.open == "thinking":
            signature = uuid4().hex
            events.append(self.delta({"type": "signature_delta", "signature": signature}))
        self.open = None
        events.append({"type": "content_block_stop", "index": self.index})
        return events

    def delta(self, delta):
        return {"type": "content_block_delta", "index": self.index, "delta": delta}

    def text(self, text):
        if self.open != "text":
            # Whitespace between blocks (after reasoning or a tool call) is not content.
            if not (self.held_whitespace + text).strip():
                self.held_whitespace += text
                return []
            text = self.held_whitespace + text
            events = self.open_block({"type": "text", "text": ""})
        else:
            events = []
        self.held_whitespace = ""
        return events + [self.delta({"type": "text_delta", "text": text})]

    def reasoning(self, text):
        if not self.show_thinking:
            return []
        self.held_whitespace = ""
        events = []
        if self.open != "thinking":
            events += self.open_block({"type": "thinking", "thinking": "", "signature": ""})
        if not self.omit_thinking:
            events.append(self.delta({"type": "thinking_delta", "thinking": text}))
        return events

    def tool_use(self, call):
        block = {
            "type": "tool_use",
            "id": "toolu_" + uuid4().hex[:24],
            "name": call.function.name,
            "input": {},
        }
        self.held_whitespace = ""
        events = self.open_block(block)
        events.append(
            self.delta({"type": "input_json_delta", "partial_json": call.function.arguments})
        )
        return events + self.close_block()

    def consume(self, packet):
        self.tool_format = packet["tool_format"]
        events = []
        for channel, text in packet["events"]:
            if not text:
                continue
            if channel == TOOL:
                # Calls are parsed once complete; later output waits so block order holds.
                self.deferred = True
            if self.deferred:
                if self.segments and self.segments[-1][0] == channel:
                    self.segments[-1][1] += text
                else:
                    self.segments.append([channel, text])
            elif channel == CONTENT:
                events += self.text(text)
            elif channel == REASONING:
                events += self.reasoning(text)
        return events

    def finish(self, generation):
        if generation.get("eos_reason") == "loop_detected":
            raise InvalidModelOutput("Generation was stopped by loop detection")
        limited = generation.get("finish_reason") == "length"
        parsed = []
        calls = []
        for channel, text in self.segments:
            found = []
            if channel == TOOL and not limited:
                # A token-limit stop may truncate a call, which is never released.
                try:
                    found = parse_toolcalls(text, self.tool_format, strict=True)
                    self.tools.validate_calls(found)
                except Exception as exc:
                    raise InvalidModelOutput(
                        f"The model produced an invalid tool call: {exc}"
                    ) from exc
                if not self.tools.parallel and len(calls) + len(found) > 1:
                    xlogger.warning("Dropping extra tool calls: parallel tool use is disabled")
                    found = found[: max(0, 1 - len(calls))]
                calls += found
            parsed.append((channel, text, found))
        if self.tools.required and not calls and not limited:
            xlogger.warning("tool_choice required a tool call, but the model made none")

        events = []
        for channel, text, found in parsed:
            if channel == CONTENT:
                events += self.text(text)
            elif channel == REASONING:
                events += self.reasoning(text)
            else:
                for call in found:
                    events += self.tool_use(call)
        events += self.close_block()

        stop_sequence = None
        if calls:
            stop_reason = "tool_use"
        elif limited:
            stop_reason = "max_tokens"
        elif (
            generation.get("eos_reason") == "stop_string"
            and generation.get("stop_str") in self.stop_sequences
        ):
            stop_reason = "stop_sequence"
            stop_sequence = generation["stop_str"]
        else:
            stop_reason = "end_turn"

        prompt_tokens = generation.get("prompt_tokens")
        if prompt_tokens is not None:
            self.message["usage"] = usage(
                prompt_tokens,
                generation.get("gen_tokens") or 0,
                round(generation.get("cached_tokens") or 0),
            )
        self.finished = True
        events.append(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                "usage": self.message["usage"],
            }
        )
        events.append({"type": "message_stop"})
        return events


def error_event(message, kind="api_error"):
    return {"type": "error", "error": {"type": kind, "message": message}}


async def generate_message_events(
    data, params, tools, prompt, embeddings, input_tokens, request_id, model_name, disconnect
):
    accumulator = MessagesAccumulator(data, tools, model_name, input_tokens)
    source = None
    try:
        source = iter_generation_events(
            request_id,
            prompt,
            params,
            _resolve_start_in_reasoning(prompt, params),
            embeddings,
            disconnect,
            f"request {request_id} messages",
        )
        for event in accumulator.start():
            yield event
        async for packet in source:
            await disconnect.poll()
            for event in accumulator.consume(packet):
                yield event
            if packet["generation"].get("finish_reason"):
                for event in accumulator.finish(packet["generation"]):
                    yield event
                break
        if not accumulator.finished:
            raise InvalidModelOutput("Generation ended without a terminal result")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        xlogger.error("Messages generation failed", {"error": str(exc)})
        message = (
            str(exc)
            if isinstance(exc, InvalidModelOutput)
            else "Generation failed; check server logs."
        )
        yield error_event(message)
    finally:
        # SSE disconnects cancel an AnyIO task group. Cleanup must survive that
        # cancellation scope so backend jobs and their cache allocations are freed.
        with anyio.CancelScope(shield=True):
            try:
                if source is not None:
                    await source.aclose()
            finally:
                await disconnect.cleanup()


async def stream_message(events):
    try:
        async for event in events:
            yield {"event": event["type"], "data": json.dumps(event, ensure_ascii=False)}
    finally:
        with anyio.CancelScope(shield=True):
            await events.aclose()


async def collect_message(events):
    """Fold the event stream into a Message, the way SDK stream helpers do."""

    message = None
    async for event in events:
        kind = event["type"]
        if kind == "error":
            raise MessageGenerationError(event["error"]["message"])
        if kind == "message_start":
            message = {**event["message"], "content": []}
        elif kind == "content_block_start":
            message["content"].append(dict(event["content_block"]))
        elif kind == "content_block_delta":
            block = message["content"][event["index"]]
            delta = event["delta"]
            if delta["type"] == "text_delta":
                block["text"] += delta["text"]
            elif delta["type"] == "thinking_delta":
                block["thinking"] += delta["thinking"]
            elif delta["type"] == "signature_delta":
                block["signature"] = delta["signature"]
            elif delta["type"] == "input_json_delta":
                block.setdefault("partial_json", "")
                block["partial_json"] += delta["partial_json"]
        elif kind == "content_block_stop":
            block = message["content"][event["index"]]
            if block["type"] == "tool_use":
                block["input"] = json.loads(block.pop("partial_json", "") or "{}")
        elif kind == "message_delta":
            message.update(event["delta"])
            message["usage"] = event["usage"]
    return message
