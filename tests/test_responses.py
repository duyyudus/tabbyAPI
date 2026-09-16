"""Responses contract tests; no model weights or GPU required."""

import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from pydantic import ValidationError

from common import model
from common.auth import check_api_key
from common.networking import add_request_id
from endpoints.OAI.responses_router import router
from endpoints.OAI.types.responses import ResponsesRequest
from endpoints.OAI.utils.chat_completion import _chat_stream_collector, iter_generation_events
from endpoints.OAI.utils.responses import ResponsesAccumulator, generate_response_events, collect_response
from endpoints.OAI.utils.responses_input import adapt_request, ResponseRequestError, InvalidModelOutput
from endpoints.OAI.utils.stream_parser import CONTENT, TOOL, REASONING


FUNCTION = {"type": "function", "name": "read", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}},
    "required": ["path"], "additionalProperties": False,
}, "strict": True}
CUSTOM = {"type": "custom", "name": "patch", "format": {
    "type": "grammar", "syntax": "lark", "definition": 'start: "*** Begin Patch" "\\n" "*** End Patch" "\\n"',
}}


def packet(events, **generation):
    return {"events": events, "generation": generation, "tool_format": "qwen3_coder"}


def tool_text(name="read", arguments=None):
    args = arguments or {"path": "x"}
    return '<tool_call><function=' + name + '>' + "".join(
        '<parameter=' + k + '>' + json.dumps(v) + '</parameter>' for k, v in args.items()
    ) + '</function></tool_call>'


def finish(**kwargs):
    return {"finish_reason": "stop", "prompt_tokens": 10, "gen_tokens": 5, "cached_tokens": 3, **kwargs}


class Disconnect:
    def __init__(self):
        self.cleaned = False

    async def poll(self):
        pass

    async def cleanup(self):
        self.cleaned = True


class FakeContainer:
    loaded = True
    harmony = False
    muse_glimmer = False
    reasoning = True
    reasoning_start_token = "<think>"
    reasoning_end_token = "</think>"
    start_in_reasoning = "never"
    tool_format = "qwen3_coder"
    tool_calls_in_reasoning = True
    use_vision = False
    model_dir = Path("/models/test-model")
    template_vars_default = {}
    template_vars_force = {}
    reasoning_budget_tokens = None
    reasoning_budget_message = None

    def __init__(self, chunks=None):
        self.chunks = chunks or [{"text": "hello"}, finish()]
        self.closed = False
        self.prompt_template = SimpleNamespace(render=self.render)
        self.hf_model = SimpleNamespace(add_bos_token=lambda: False)

    async def render(self, variables):
        return json.dumps(variables["messages"])

    def get_special_tokens(self):
        return {}

    async def stream_generate(self, *args, **kwargs):
        try:
            for chunk in self.chunks:
                if isinstance(chunk, Exception):
                    raise chunk
                yield dict(chunk)
        finally:
            self.closed = True

    def check_context_length(self, *args, **kwargs):
        pass


def make_app():
    app = FastAPI(dependencies=[])
    app.include_router(router)
    async def no_auth():
        pass
    app.dependency_overrides[check_api_key] = no_auth

    @app.middleware("http")
    async def request_id(request, call_next):
        await add_request_id(request)
        return await call_next(request)
    return app


class InputTests(unittest.TestCase):
    def test_instructions_shorthand_and_sampling(self):
        data = ResponsesRequest(input=[{"role": "user", "content": "hello"}],
                                instructions="help", max_output_tokens=42, temperature=0.2)
        chat, _ = adapt_request(data)
        self.assertEqual([m.content for m in chat.messages], ["help", "hello"])
        self.assertEqual(chat.max_tokens, 42)
        self.assertTrue(chat._fail_on_grammar_error)

    def test_function_custom_replay_and_parallel_results(self):
        raw = "  *** Begin Patch\n*** End Patch\n"
        data = ResponsesRequest(input=[
            {"role": "assistant", "content": [{"type": "output_text", "text": "checking"}]},
            {"type": "function_call", "call_id": "one", "name": "read", "arguments": '{"path":"x"}'},
            {"type": "custom_tool_call", "call_id": "two", "name": "patch", "input": raw},
            {"type": "custom_tool_call_output", "call_id": "two", "output": "ok"},
            {"type": "function_call_output", "call_id": "one", "output": [{"type": "input_text", "text": "x"}]},
        ])
        chat, _ = adapt_request(data)
        self.assertEqual(len(chat.messages[0].tool_calls), 2)
        self.assertEqual(json.loads(chat.messages[0].tool_calls[1].function.arguments)["input"], raw)
        self.assertEqual(chat.messages[1].tool_call_id, "two")

    def test_invalid_replay(self):
        for items in [
            [{"type": "function_call_output", "call_id": "missing", "output": "x"}],
            [{"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"}],
        ]:
            with self.assertRaises(ResponseRequestError):
                adapt_request(ResponsesRequest(input=items))

    def test_unsupported_options(self):
        for option in ({"store": True}, {"previous_response_id": "r"}, {"background": True},
                       {"conversation": "c"}, {"unknown": True}, {"reasoning": {"summary": "auto"}},
                       {"tools": [{"type": "web_search"}]}):
            with self.assertRaises(ValidationError):
                ResponsesRequest(input="x", **option)

    def test_images_rejected_on_text_model(self):
        data = ResponsesRequest(input=[{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,eA=="},
        ]}])
        with self.assertRaises(ResponseRequestError):
            adapt_request(data)
        chat, _ = adapt_request(data, vision=True)
        self.assertEqual(chat.messages[0].content[0].type, "image_url")

    def test_schema_and_grammar_preflight(self):
        for tool in [
            {**FUNCTION, "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}},
            {**CUSTOM, "format": {"type": "grammar", "syntax": "lark", "definition": "%import secrets.VALUE\nstart: VALUE"}},
            {**FUNCTION, "parameters": {"$ref": "https://example.com/schema"}},
        ]:
            with self.assertRaises(ResponseRequestError):
                adapt_request(ResponsesRequest(input="x", tools=[tool]))

    def test_normalized_strict_echo(self):
        data = ResponsesRequest(input="x", tools=[{"type": "function", "name": "f", "parameters": {
            "type": "object", "properties": {"x": {"type": "string"}},
        }}])
        adapt_request(data)
        self.assertTrue(data.tools[0].strict)
        self.assertEqual(data.tools[0].parameters["required"], ["x"])


class AccumulatorTests(unittest.TestCase):
    def make(self, **kwargs):
        data = ResponsesRequest(input="hello", **kwargs)
        _, tools = adapt_request(data)
        return ResponsesAccumulator(data, tools, "test-model")

    def test_text_lifecycle_usage_and_snapshot(self):
        acc = self.make()
        events = acc.start()
        events += acc.consume(packet([(CONTENT, "hello")]))
        events += acc.finish(finish())
        self.assertEqual(events[0]["response"]["output"], [])
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
        self.assertEqual(acc.response.usage.total_tokens, 15)
        self.assertIsNone(acc.response.usage.output_tokens_details.reasoning_tokens)
        self.assertEqual(acc.response.output[0].content[0].text, "hello")

    def test_mixed_order_custom_input(self):
        acc = self.make(tools=[FUNCTION, CUSTOM])
        value = "*** Begin Patch\n*** End Patch\n"
        acc.consume(packet([(CONTENT, "before"), (TOOL, tool_text()), (REASONING, "hidden"),
                            (CONTENT, "between"), (TOOL, tool_text("patch", {"input": value})),
                            (CONTENT, "after")]))
        acc.finish(finish())
        self.assertEqual([i.type for i in acc.response.output],
                         ["message", "function_call", "message", "custom_tool_call", "message"])
        self.assertEqual(acc.response.output[3].input, value)
        self.assertNotIn("hidden", acc.response.model_dump_json())

    def test_limit_never_releases_tool(self):
        acc = self.make(tools=[FUNCTION], tool_choice="required")
        acc.consume(packet([(CONTENT, "before"), (TOOL, '<tool_call>{"name":')]))
        acc.finish(finish(finish_reason="length"))
        self.assertEqual(acc.response.status, "incomplete")
        self.assertFalse(any(i.type == "function_call" for i in acc.response.output))
        self.assertEqual(acc.response.incomplete_details["reason"], "max_output_tokens")

    def test_bad_calls_fail_before_any_tool_emitted(self):
        for options, text in [
            ({"tools": [FUNCTION]}, tool_text("missing")),
            ({"tools": [FUNCTION]}, tool_text(arguments={"path": 4})),
            ({"tools": [FUNCTION]}, '<tool_call>nonsense</tool_call>'),
            ({"tools": [CUSTOM]}, tool_text("patch", {"input": "invalid"})),
            ({"tools": [FUNCTION], "parallel_tool_calls": False}, tool_text() + tool_text()),
            ({"tools": [FUNCTION], "tool_choice": "none"}, tool_text()),
        ]:
            acc = self.make(**options)
            acc.consume(packet([(TOOL, text)]))
            with self.assertRaises((InvalidModelOutput, ValueError)):
                acc.finish(finish())
            self.assertFalse(acc.response.output)

    def test_required_and_named_selection(self):
        acc = self.make(tools=[FUNCTION], tool_choice={"type": "function", "name": "read"})
        acc.consume(packet([(CONTENT, "no tool")]))
        with self.assertRaises(InvalidModelOutput):
            acc.finish(finish())

    def test_structured_output(self):
        acc = self.make(text={"format": {"type": "json_object"}})
        acc.consume(packet([(CONTENT, '{"ok":true}')]))
        acc.finish(finish())
        bad = self.make(text={"format": {"type": "json_object"}})
        bad.consume(packet([(CONTENT, "not json")]))
        with self.assertRaises(InvalidModelOutput):
            bad.finish(finish())


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_order_and_unmodified_length_stop(self):
        container = FakeContainer([{"text": "<think>hmm</think>hello" + tool_text()},
                                   finish(finish_reason="length")])
        params, _ = adapt_request(ResponsesRequest(input="x"))
        with patch.object(model, "container", container):
            packets = [p async for p in iter_generation_events("id", "prompt", params, False)]
        self.assertEqual([c for c, _ in packets[0]["events"]], [REASONING, CONTENT, TOOL])
        self.assertEqual(packets[-1]["generation"]["finish_reason"], "length")
        self.assertTrue(container.closed)

    async def test_chat_collector_regression(self):
        params, _ = adapt_request(ResponsesRequest(input="x"))
        with patch.object(model, "container", FakeContainer([{"text": "<think>hmm</think>hello"}, finish()])):
            result = await _chat_stream_collector(0, None, "id", "p", params, False, streaming_mode=False)
        self.assertEqual(result["content"], "hello")
        self.assertEqual(result["reasoning_content"], "hmm")
        self.assertEqual(result["finish_reason"], "stop")

    async def test_cleanup_on_backend_failure(self):
        data = ResponsesRequest(input="x")
        params, tools = adapt_request(data)
        disconnect = Disconnect()
        container = FakeContainer([RuntimeError("broken")])
        with patch.object(model, "container", container):
            result = await collect_response(generate_response_events(data, params, tools, "p", None,
                                                                     "id", "test", disconnect))
        self.assertEqual(result["status"], "failed")
        self.assertTrue(disconnect.cleaned)
        self.assertTrue(container.closed)

    async def test_cleanup_on_cancel(self):
        data = ResponsesRequest(input="x")
        params, tools = adapt_request(data)
        disconnect = Disconnect()
        container = FakeContainer()
        with patch.object(model, "container", container):
            events = generate_response_events(data, params, tools, "p", None, "id", "test", disconnect)
            await anext(events)
            await anext(events)
            await anext(events)
            await events.aclose()
        self.assertTrue(disconnect.cleaned)
        self.assertTrue(container.closed)


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.container = FakeContainer()
        self.patcher = patch.object(model, "container", self.container)
        self.patcher.start()
        self.context_patch = patch.object(model, "check_context_length")
        self.context_patch.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app()), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.patcher.stop()
        self.context_patch.stop()

    async def test_json_and_sse_agree(self):
        response = await self.client.post("/v1/responses", json={"input": "hi"})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        streamed = await self.client.post("/v1/responses", json={"input": "hi", "stream": True})
        self.assertIn("event: response.completed", streamed.text)
        events = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(events[-1]["response"]["output"][0]["content"], result["output"][0]["content"])
        self.assertEqual(events[-1]["response"]["usage"], result["usage"])
        self.assertNotIn("[DONE]", streamed.text)

    async def test_validation_error_shape(self):
        response = await self.client.post("/v1/responses", json={"input": "x", "store": True})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "store")

    async def test_openai_sdk_create_and_stream(self):
        from openai import AsyncOpenAI
        sdk = AsyncOpenAI(api_key="test", base_url="http://test/v1", http_client=self.client, max_retries=0)
        result = await sdk.responses.create(input="hi", model="test-model", store=False)
        self.assertEqual(result.output_text, "hello")
        stream = await sdk.responses.create(input="hi", model="test-model", stream=True, store=False)
        events = [event async for event in stream]
        self.assertEqual(events[-1].type, "response.completed")
        self.assertEqual(events[-1].response.output_text, "hello")


class AdditionalContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_python_sdk_tool_round_trips(self):
        from openai import AsyncOpenAI
        for tool in (FUNCTION, CUSTOM):
            value = "*** Begin Patch\n*** End Patch\n"
            name = tool["name"]
            args = {"path": "x"} if name == "read" else {"input": value}
            container = FakeContainer([{"text": tool_text(name, args)}, finish()])
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app())) as client:
                sdk = AsyncOpenAI(api_key="test", base_url="http://test/v1", http_client=client, max_retries=0)
                with patch.object(model, "container", container), patch.object(model, "check_context_length"):
                    stream = await sdk.responses.create(model="test-model", input="use tool", tools=[tool], stream=True)
                    events = [e async for e in stream]
                    response = events[-1].response
                    self.assertEqual(response.status, "completed")
                    call = response.output[0]
                    self.assertEqual(call.type, "function_call" if name == "read" else "custom_tool_call")
                    if name == "patch":
                        self.assertEqual(call.input, value)
                    container.chunks = [{"text": "done"}, finish()]
                    # SDK dumps contain optional nulls on models: use exclude_none when replaying.
                    input_items = [{"role": "user", "content": "use tool"},
                                   call.model_dump(exclude_none=True),
                                   {"type": call.type + "_output", "call_id": call.call_id, "output": "ok"}]
                    result = await sdk.responses.create(model="test-model", input=input_items, tools=[tool])
                    self.assertEqual(result.output_text, "done")

    async def test_auth_and_context_errors_are_openai_shaped(self):
        from fastapi import HTTPException
        from common.errors import ContextLengthHTTPException
        app = make_app()
        async def unauthorized():
            raise HTTPException(401, "Invalid API key")
        app.dependency_overrides[check_api_key] = unauthorized
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/responses", json={"input": "hi"})
            self.assertEqual(response.status_code, 401)
            self.assertIn("error", response.json())
        with patch.object(model, "container", FakeContainer()), patch.object(
            model, "check_context_length", side_effect=ContextLengthHTTPException("too long")
        ):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app()), base_url="http://test") as client:
                response = await client.post("/v1/responses", json={"input": "hi", "stream": True})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["error"]["code"], "context_length_exceeded")
                self.assertNotIn("text/event-stream", response.headers["content-type"])

    async def test_model_failure_after_stream_started(self):
        with patch.object(model, "container", FakeContainer([{"text": "partial"}, RuntimeError("bad")])), patch.object(model, "check_context_length"):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app()), base_url="http://test") as client:
                response = await client.post("/v1/responses", json={"input": "hi", "stream": True})
                self.assertIn("event: response.failed", response.text)
                self.assertNotIn("event: response.completed", response.text)

    async def test_client_disconnect_runs_cleanup(self):
        class Disconnected(Disconnect):
            async def poll(self):
                raise asyncio.CancelledError()
        data = ResponsesRequest(input="hi")
        params, tools = adapt_request(data)
        disconnect = Disconnected()
        container = FakeContainer()
        with patch.object(model, "container", container):
            with self.assertRaises(asyncio.CancelledError):
                await collect_response(generate_response_events(data, params, tools, "p", None, "id", "test", disconnect))
        self.assertTrue(disconnect.cleaned)
        self.assertTrue(container.closed)

    def test_partial_parser_success_is_rejected(self):
        data = ResponsesRequest(input="x", tools=[FUNCTION])
        _, tools = adapt_request(data)
        acc = ResponsesAccumulator(data, tools, "test")
        acc.consume(packet([(TOOL, tool_text() + '<tool_call>broken</tool_call>')]))
        with self.assertRaises(InvalidModelOutput):
            acc.finish(finish())
        self.assertFalse(acc.response.output)

    def test_allowed_tools_and_strict_fallback(self):
        data = ResponsesRequest(input="x", tools=[FUNCTION, {"type": "function", "name": "loose", "parameters": {
            "type": "object", "properties": {}, "additionalProperties": True,
        }}], tool_choice={"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "read"}]})
        chat, tools = adapt_request(data)
        self.assertEqual(tools.allowed, {"read"})
        self.assertEqual(len(chat.tools), 1)
        self.assertFalse(data.tools[1].strict)

    def test_unknown_strict_constraints_rejected(self):
        data = ResponsesRequest(input="x", tools=[{**FUNCTION, "parameters": {
            **FUNCTION["parameters"], "properties": {"path": {"type": "string", "format": "email"}},
        }}])
        with self.assertRaises(ResponseRequestError):
            adapt_request(data)

    def test_no_fabricated_reasoning_usage(self):
        data = ResponsesRequest(input="x")
        _, tools = adapt_request(data)
        acc = ResponsesAccumulator(data, tools, "test")
        acc.consume(packet([(REASONING, "thinking"), (CONTENT, "answer")]))
        acc.finish(finish())
        self.assertNotIn("output_tokens_details", acc.snapshot()["usage"])

    def test_grammar_failure_is_not_silently_ignored(self):
        import importlib.util
        import sys
        fake_exl = SimpleNamespace(Tokenizer=object, Filter=object, LLGuidanceFilter=None)
        spec = importlib.util.spec_from_file_location("grammar_under_test", "backends/exllamav3/grammar.py")
        grammar = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"exllamav3": fake_exl}):
            spec.loader.exec_module(grammar)
        def fail(*args, **kwargs):
            raise ValueError("bad grammar")
        grammar.LLGuidanceFilter = fail
        with self.assertRaises(ValueError):
            grammar.ExLlamaV3Grammar().add_json_schema_filter({}, None, fail_on_error=True)


class CancellationScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_sse_style_cancel_scope_allows_async_cleanup(self):
        import anyio
        started = asyncio.Event()
        class WaitingContainer(FakeContainer):
            async def stream_generate(self, *args, **kwargs):
                try:
                    started.set()
                    await anyio.sleep_forever()
                    yield finish()
                finally:
                    self.closed = True
        class AsyncCleanup(Disconnect):
            async def cleanup(self):
                await anyio.sleep(0)
                self.cleaned = True
        data = ResponsesRequest(input="hi")
        params, tools = adapt_request(data)
        disconnect = AsyncCleanup()
        container = WaitingContainer()
        async def consume():
            await collect_response(generate_response_events(data, params, tools, "p", None, "id", "test", disconnect))
        with patch.object(model, "container", container):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(consume)
                await started.wait()
                tasks.cancel_scope.cancel()
        self.assertTrue(container.closed)
        self.assertTrue(disconnect.cleaned)
