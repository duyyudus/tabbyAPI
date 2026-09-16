"""Responses accumulation and serialization over the shared generation iterator."""

import asyncio
import json

import anyio
from time import time
from uuid import uuid4

from common.logger import xlogger
from endpoints.OAI.types.responses import (
    CustomCall,
    CustomTool,
    FunctionCall,
    InputTokenDetails,
    OutputMessage,
    OutputText,
    ResponseEvent,
    ResponseObject,
    ResponseUsage,
)
from endpoints.OAI.utils.chat_completion import iter_generation_events, _resolve_start_in_reasoning
from endpoints.OAI.utils.responses_input import InvalidModelOutput, schema_validator
from endpoints.OAI.utils.stream_parser import CONTENT, TOOL
from endpoints.OAI.utils.tools import parse_toolcalls


class ResponsesAccumulator:
    def __init__(self, data, tools, model_name):
        self.response = ResponseObject(
            model=model_name,
            **data.model_dump(
                include={
                    "instructions",
                    "max_output_tokens",
                    "temperature",
                    "top_p",
                    "parallel_tool_calls",
                    "tool_choice",
                    "tools",
                    "text",
                    "reasoning",
                    "metadata",
                },
                by_alias=True,
            ),
        )
        self.tools = tools
        self.sequence = 0
        self.message = None
        self.segments = []
        self.last_channel = None
        self.deferred = False
        self.tool_format = None
        self.finished = False

    def event(self, kind, **payload):
        event = ResponseEvent(type=kind, sequence_number=self.sequence, **payload)
        self.sequence += 1
        # Snapshot now: later mutation must not change already emitted events.
        return event.model_dump(mode="json", by_alias=True)

    def snapshot(self):
        return self.response.model_dump(mode="json", by_alias=True)

    def start(self):
        return [
            self.event("response.created", response=self.snapshot()),
            self.event("response.in_progress", response=self.snapshot()),
        ]

    def text(self, text):
        events = []
        if self.message is None:
            self.message = OutputMessage()
            self.response.output.append(self.message)
            index = len(self.response.output) - 1
            events.append(
                self.event(
                    "response.output_item.added", output_index=index, item=self.message.model_dump()
                )
            )
            self.message.content.append(OutputText())
            events.append(
                self.event(
                    "response.content_part.added",
                    output_index=index,
                    item_id=self.message.id,
                    content_index=0,
                    part=self.message.content[0].model_dump(),
                )
            )
        self.message.content[0].text += text
        events.append(
            self.event(
                "response.output_text.delta",
                output_index=len(self.response.output) - 1,
                item_id=self.message.id,
                content_index=0,
                delta=text,
                logprobs=[],
            )
        )
        return events

    def close_text(self, status="completed"):
        if self.message is None:
            return []
        message = self.message
        self.message = None
        message.status = status
        common = {
            "output_index": len(self.response.output) - 1,
            "item_id": message.id,
            "content_index": 0,
        }
        return [
            self.event(
                "response.output_text.done", **common, text=message.content[0].text, logprobs=[]
            ),
            self.event(
                "response.content_part.done", **common, part=message.content[0].model_dump()
            ),
            self.event(
                "response.output_item.done",
                output_index=common["output_index"],
                item=message.model_dump(),
            ),
        ]

    def consume(self, packet):
        self.tool_format = packet["tool_format"]
        events = []
        for channel, text in packet["events"]:
            if not text:
                continue
            if channel != CONTENT and self.last_channel == CONTENT and not self.deferred:
                events += self.close_text()
            if channel == TOOL:
                self.deferred = True
            if self.deferred:
                if self.segments and self.segments[-1][0] == channel:
                    self.segments[-1][1] += text
                else:
                    self.segments.append([channel, text])
            elif channel == CONTENT:
                events += self.text(text)
            self.last_channel = channel
        return events

    def emit_call(self, call):
        name = call.function.name
        custom = isinstance(self.tools.tools[name], CustomTool)
        value = json.loads(call.function.arguments)["input"] if custom else call.function.arguments
        common = {
            "id": ("ctc_" if custom else "fc_") + uuid4().hex,
            "call_id": "call_" + uuid4().hex,
            "name": name,
            "status": "in_progress",
        }
        item = CustomCall(**common, input="") if custom else FunctionCall(**common, arguments="")
        index = len(self.response.output)
        self.response.output.append(item)
        events = [
            self.event("response.output_item.added", output_index=index, item=item.model_dump())
        ]
        prefix = "response.custom_tool_call_input" if custom else "response.function_call_arguments"
        events.append(
            self.event(prefix + ".delta", output_index=index, item_id=item.id, delta=value)
        )
        if custom:
            item.input = value
        else:
            item.arguments = value
        events.append(
            self.event(
                prefix + ".done",
                output_index=index,
                item_id=item.id,
                **({"input": value} if custom else {"arguments": value, "name": name}),
            )
        )
        item.status = "completed"
        events.append(
            self.event("response.output_item.done", output_index=index, item=item.model_dump())
        )
        return events

    def finish(self, generation):
        events = []
        prompt_tokens = generation.get("prompt_tokens")
        output_tokens = generation.get("gen_tokens")
        if prompt_tokens is not None and output_tokens is not None:
            self.response.usage = ResponseUsage(
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=prompt_tokens + output_tokens,
                input_tokens_details=InputTokenDetails(
                    cached_tokens=round(generation.get("cached_tokens") or 0)
                ),
            )
        limited = generation.get("finish_reason") == "length"
        if generation.get("eos_reason") == "loop_detected":
            raise InvalidModelOutput("Generation was stopped by loop detection")
        # Never release a possibly truncated executable call on a token-limit stop.
        parsed_segments = []
        if not limited:
            calls = []
            for channel, text in self.segments:
                parsed = None
                if channel == TOOL:
                    try:
                        parsed = parse_toolcalls(text, self.tool_format, strict=True)
                    except Exception as exc:
                        raise InvalidModelOutput(
                            "The model produced an unparseable tool call"
                        ) from exc
                    calls.extend(parsed)
                parsed_segments.append((channel, text, parsed))
            self.tools.validate_calls(calls)
            fmt = self.response.text.format
            if fmt.type != "text":
                text = "".join(
                    part.text
                    for item in self.response.output
                    if isinstance(item, OutputMessage)
                    for part in item.content
                )
                try:
                    value = json.loads(text)
                    schema_validator(
                        fmt.schema_ if fmt.type == "json_schema" else {"type": "object"}
                    ).validate(value)
                except Exception as exc:
                    raise InvalidModelOutput(
                        "Generated text does not satisfy the requested JSON format"
                    ) from exc
        else:
            parsed_segments = [(c, t, None) for c, t in self.segments]
        for channel, text, calls in parsed_segments:
            if channel == CONTENT:
                events += self.text(text)
            else:
                events += self.close_text("incomplete" if limited else "completed")
                for call in calls or []:
                    events += self.emit_call(call)
        events += self.close_text("incomplete" if limited else "completed")
        self.response.status = "incomplete" if limited else "completed"
        if limited:
            self.response.incomplete_details = {"reason": "max_output_tokens"}
        else:
            self.response.completed_at = int(time())
        self.finished = True
        events.append(self.event("response." + self.response.status, response=self.snapshot()))
        return events

    def fail(self, message, code="server_error"):
        events = self.close_text("incomplete")
        self.response.status = "failed"
        self.response.error = {"code": code, "message": message}
        self.finished = True
        events.append(self.event("response.failed", response=self.snapshot()))
        return events


async def generate_response_events(
    data, params, tools, prompt, embeddings, request_id, model_name, disconnect_handler
):
    accumulator = ResponsesAccumulator(data, tools, model_name)
    source = None
    try:
        source = iter_generation_events(
            request_id,
            prompt,
            params,
            _resolve_start_in_reasoning(prompt, params),
            embeddings,
            disconnect_handler,
            f"request {request_id} responses",
        )
        for event in accumulator.start():
            yield event
        async for packet in source:
            await disconnect_handler.poll()
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
        xlogger.error("Responses generation failed", {"error": str(exc)})
        message = (
            str(exc)
            if isinstance(exc, InvalidModelOutput)
            else "Generation failed; check server logs."
        )
        for event in accumulator.fail(message):
            yield event
    finally:
        # SSE disconnects cancel an AnyIO task group. Cleanup must survive that
        # cancellation scope so backend jobs and their cache allocations are freed.
        with anyio.CancelScope(shield=True):
            try:
                if source is not None:
                    await source.aclose()
            finally:
                await disconnect_handler.cleanup()


async def stream_response(events):
    try:
        async for event in events:
            yield {"event": event["type"], "data": json.dumps(event, ensure_ascii=False)}
    finally:
        with anyio.CancelScope(shield=True):
            await events.aclose()


async def collect_response(events):
    result = None
    async for event in events:
        if event["type"] in {"response.completed", "response.incomplete", "response.failed"}:
            result = event["response"]
    return result
