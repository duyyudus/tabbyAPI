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
from endpoints.OAI.utils import responses_input
from endpoints.OAI.utils.stream_parser import CONTENT, TOOL, REASONING


FUNCTION = {"type": "function", "name": "read", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}},
    "required": ["path"], "additionalProperties": False,
}, "strict": True}
CUSTOM = {"type": "custom", "name": "patch", "format": {
    "type": "grammar", "syntax": "lark", "definition": 'start: "*** Begin Patch" "\\n" "*** End Patch" "\\n"',
}}
NAMESPACE = {
    "type": "namespace",
    "name": "workspace",
    "description": "Tools for working with the client workspace.",
    "tools": [
        {**FUNCTION, "description": "Read a workspace file."},
        {**CUSTOM, "description": "Apply a patch to the workspace."},
    ],
}
CODEX_NAMESPACE = {
    "type": "namespace",
    "name": "multi_agent_v1",
    "description": "Tools for spawning and managing sub-agents.",
    "tools": [{
        "type": "function",
        "name": "spawn_agent",
        "description": "Spawn a sub-agent for a well-scoped task.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
    }],
}


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
                       {"conversation": "c"}, {"unknown": True},
                       {"prompt": {"id": "p"}}, {"max_tool_calls": 3},
                       {"context_management": [{"type": "compaction", "compact_threshold": 1000}]},
                       {"truncation": "auto"},
                       {"reasoning": {"summary": "verbose"}},
                       {"tools": [{"type": "function"}]}):
            with self.assertRaises(ValidationError):
                ResponsesRequest(input="x", **option)

    def test_official_request_limits(self):
        for option in (
            {"max_output_tokens": 15},
            {"metadata": {"k" * 65: "value"}},
            {"metadata": {"key": "v" * 513}},
            {"metadata": {str(index): "value" for index in range(17)}},
        ):
            with self.subTest(option=option), self.assertRaises(ValidationError):
                ResponsesRequest(input="x", **option)
        ResponsesRequest(
            input="x", max_output_tokens=16, metadata={"k" * 64: "v" * 512}
        )

    def test_unsupported_options_state_a_reason(self):
        """A rejection that cannot be acted on is a debugging session, not an error."""
        for option, expected in (
            ({"store": True}, "stateless"),
            ({"previous_response_id": "r"}, "replay the earlier items"),
            ({"truncation": "auto"}, "truncated"),
            ({"prompt": {"id": "p"}}, "instructions instead"),
            ({"context_management": [{"type": "compaction"}]}, "not implemented"),
            ({"moderation": {"model": "m"}}, "filtering that never happens"),
        ):
            with self.assertRaises(ValidationError) as caught:
                ResponsesRequest(input="x", **option)
            self.assertIn(expected, caught.exception.errors()[0]["msg"])

    def test_data_requests_accepted_but_unproduced(self):
        """An include asks for available data; none of it exists, and none is invented."""
        data = ResponsesRequest(
            input="x", top_logprobs=5,
            include=["reasoning.encrypted_content", "message.output_text.logprobs",
                     "web_search_call.results"],
        )
        self.assertEqual(len(data.include), 3)
        chat, _ = adapt_request(data)
        self.assertEqual([m.content for m in chat.messages], ["x"])

    def test_advisory_fields_accepted_and_ignored(self):
        """Routing and telemetry hints cannot change generation, so they must not 400."""
        data = ResponsesRequest(input="x", service_tier="priority", user="u1",
                                safety_identifier="s1", prompt_cache_key="k",
                                prompt_cache_retention="24h",
                                prompt_cache_options={"mode": "explicit"},
                                stream_options={"include_obfuscation": False},
                                client_metadata={"app": "codex"})
        self.assertEqual(data.service_tier, "priority")
        chat, _ = adapt_request(data)
        self.assertEqual([m.content for m in chat.messages], ["x"])

    def test_moderation_still_rejected(self):
        """Ignoring a moderation request would imply filtering that never happens."""
        with self.assertRaises(ValidationError):
            ResponsesRequest(input="x", moderation={"model": "omni-moderation-latest"})

    def test_null_tool_fields_take_their_defaults(self):
        """Clients spell an unset description or parameter set as an explicit null."""
        data = ResponsesRequest(input="x", tools=[
            {"type": "function", "name": "f", "description": None, "parameters": None},
            {"type": "custom", "name": "c", "description": None},
        ])
        self.assertEqual(data.tools[0].description, "")
        self.assertEqual(data.tools[0].parameters, {"type": "object", "properties": {}})
        self.assertEqual(data.tools[1].description, "")
        chat, tools = adapt_request(data)
        self.assertEqual(set(tools.tools), {"f", "c"})

    def test_replayed_output_text_extras_are_dropped(self):
        """An assistant turn recorded elsewhere replays without its annotations."""
        data = ResponsesRequest(input=[{"role": "assistant", "content": [{
            "type": "output_text", "text": "hi",
            "annotations": [{"type": "url_citation", "url": "https://a"}],
            "logprobs": [{"token": "hi", "logprob": -0.1}],
        }]}])
        part = data.input[0].content[0]
        self.assertEqual((part.annotations, part.logprobs), ([], []))
        chat, _ = adapt_request(data)
        self.assertNotIn("url_citation", json.dumps(
            [m.model_dump() for m in chat.messages], default=str))

    def test_image_detail_hint_accepted_and_ignored(self):
        """Resolution control is unimplemented; the hint must not fail the request."""
        for detail in ("auto", "low", "high", None):
            data = ResponsesRequest(input=[{"role": "user", "content": [
                {"type": "input_image", "image_url": "https://a/b.png", "detail": detail}]}])
            chat, _ = adapt_request(data, vision=True)
            self.assertEqual(chat.messages[0].content[0].type, "image_url")

    def test_reasoning_summary_requested_but_never_produced(self):
        """Agent clients ask for summaries; accepting the ask produces no summary."""
        for summary in ("auto", "concise", "detailed", "none", None):
            data = ResponsesRequest(input="x", reasoning={"effort": "high", "summary": summary})
            self.assertEqual(data.reasoning.summary, summary)
            chat, _ = adapt_request(data)
            self.assertEqual(chat.reasoning_effort, "high")

    def test_replayed_reasoning_items_are_ignored(self):
        """Codex echoes reasoning items back; they must not break call/result pairing."""
        data = ResponsesRequest(input=[
            {"role": "user", "content": "hi"},
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque",
             "summary": [{"type": "summary_text", "text": "ignored"}]},
            {"type": "function_call", "call_id": "c1", "name": "read", "arguments": '{"path": "f"}'},
            {"type": "reasoning", "id": "rs_2"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ], tools=[FUNCTION])
        chat, _ = adapt_request(data)
        self.assertEqual([m.role for m in chat.messages], ["user", "assistant", "tool"])
        self.assertNotIn("opaque", json.dumps([m.model_dump() for m in chat.messages], default=str))

    def test_hosted_tools_accepted_but_ignored(self):
        data = ResponsesRequest(input="x", tools=[
            {"type": "web_search"}, FUNCTION,
            {"type": "turbo_search", "filters": {"allowed_domains": ["example.com"]}},
        ])
        chat, tools = adapt_request(data)
        self.assertEqual([tool.function.name for tool in chat.tools], ["read"])
        self.assertEqual(set(tools.tools), {"read"})
        self.assertEqual(tools.ignored_types, ["web_search", "turbo_search"])
        dumped = data.model_dump(mode="json")
        self.assertEqual(dumped["tools"][0], {"type": "web_search"})
        self.assertEqual(dumped["tools"][2]["type"], "turbo_search")

    def test_ignored_tools_are_named_once_per_tool_set(self):
        """The console names each ignored tool, never its configuration or credentials."""
        mcp = {"type": "mcp", "server_label": "github",
               "server_url": "https://example.com/mcp?token=URL_SECRET",
               "headers": {"Authorization": "Bearer HEADER_SECRET"}}
        tools = [{"type": "web_search"}, FUNCTION, mcp, {"type": "web_search"}]
        with patch.object(responses_input, "_warned_tool_sets", set()), \
                patch.object(responses_input, "xlogger") as log:
            adapt_request(ResponsesRequest(input="x", tools=tools))
            # Agent clients resend their tools every turn, possibly reordered.
            adapt_request(ResponsesRequest(input="x", tools=list(reversed(tools))))
            adapt_request(ResponsesRequest(input="x", tools=[{"type": "file_search"}]))
        warnings = [call.kwargs["details"] for call in log.warning.call_args_list]
        self.assertEqual(warnings, ["web_search(x2), mcp:github", "file_search"])
        self.assertEqual(log.debug.call_count, 1)
        self.assertNotIn("SECRET", json.dumps(log.method_calls, default=str))

    def test_ignored_tool_names_cannot_break_the_log_line(self):
        self.assertEqual(responses_input.loggable("evil\n12:00 ERROR:\x1b[31m x"),
                         "evil 12:00 ERROR: [31m x")
        self.assertEqual(len(responses_input.loggable("a" * 500)), 64)
        with patch.object(responses_input, "_warned_tool_sets", set()), \
                patch.object(responses_input, "xlogger") as log:
            adapt_request(ResponsesRequest(input="x", tools=[{"type": "evil\n12:00 ERROR"}]))
        self.assertNotIn("\n", log.warning.call_args.args[1]["types"])

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

    def test_namespace_tools_are_qualified_and_preserved(self):
        data = ResponsesRequest(input="x", tools=[NAMESPACE])
        chat, tools = adapt_request(data)
        self.assertEqual(set(tools.tools), {"workspace.read", "workspace.patch"})
        self.assertEqual(
            [tool.function.name for tool in chat.tools],
            ["workspace.read", "workspace.patch"],
        )
        self.assertTrue(
            chat.tools[0].function.description.startswith(
                "Tools for working with the client workspace.\n"
            )
        )
        dumped = data.model_dump(mode="json")
        self.assertEqual(dumped["tools"][0]["type"], "namespace")
        self.assertEqual(
            [tool["name"] for tool in dumped["tools"][0]["tools"]],
            ["read", "patch"],
        )

    def test_codex_multi_agent_namespace_payload(self):
        data = ResponsesRequest(input="x", tools=[CODEX_NAMESPACE])
        chat, tools = adapt_request(data)
        self.assertEqual(set(tools.tools), {"multi_agent_v1.spawn_agent"})
        self.assertEqual(chat.tools[0].function.name, "multi_agent_v1.spawn_agent")
        self.assertFalse(data.tools[0].tools[0].strict)

    def test_namespace_qualified_tool_choice(self):
        data = ResponsesRequest(
            input="x",
            tools=[NAMESPACE],
            tool_choice={"type": "function", "name": "workspace.read"},
        )
        chat, tools = adapt_request(data)
        self.assertEqual(tools.allowed, {"workspace.read"})
        self.assertEqual([tool.function.name for tool in chat.tools], ["workspace.read"])

    def test_namespace_name_collisions_rejected(self):
        for declared in [
            [NAMESPACE, {**FUNCTION, "name": "workspace.read"}],
            [{**NAMESPACE, "tools": [FUNCTION, FUNCTION]}],
        ]:
            with self.assertRaises(ResponseRequestError):
                adapt_request(ResponsesRequest(input="x", tools=declared))

    def test_namespace_rejects_nested_namespaces_and_execution_metadata(self):
        nested = {
            "type": "namespace",
            "name": "outer",
            "tools": [{"type": "namespace", "name": "inner", "tools": []}],
        }
        unsupported = [
            {**FUNCTION, "defer_loading": True},
            {**FUNCTION, "async": True},
            {**FUNCTION, "allowed_callers": ["direct"]},
            {**FUNCTION, "output_schema": {"type": "object"}},
        ]
        with self.assertRaises(ValidationError):
            ResponsesRequest(input="x", tools=[nested])
        for child in unsupported:
            with self.assertRaises(ValidationError):
                ResponsesRequest(
                    input="x",
                    tools=[{"type": "namespace", "name": "ns", "tools": [child]}],
                )

    def test_qualified_calls_replay(self):
        data = ResponsesRequest(input=[
            {
                "type": "function_call",
                "call_id": "one",
                "name": "workspace.read",
                "arguments": '{"path":"x"}',
            },
            {"type": "function_call_output", "call_id": "one", "output": "ok"},
        ])
        chat, _ = adapt_request(data)
        self.assertEqual(chat.messages[0].tool_calls[0].function.name, "workspace.read")


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

    def test_stream_item_lifecycle_and_ids(self):
        acc = self.make(tools=[FUNCTION])
        events = acc.start()
        events += acc.consume(packet([(CONTENT, "before"), (TOOL, tool_text())]))
        events += acc.finish(finish())
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds[:2], ["response.created", "response.in_progress"])
        self.assertEqual(kinds[-1], "response.completed")
        self.assertEqual(
            [event["item"]["type"] for event in events if event["type"] == "response.output_item.done"],
            ["message", "function_call"],
        )
        for index, item in enumerate(events[-1]["response"]["output"]):
            added = next(event for event in events if event["type"] == "response.output_item.added"
                         and event["output_index"] == index)
            done = next(event for event in events if event["type"] == "response.output_item.done"
                        and event["output_index"] == index)
            self.assertEqual(added["item"]["id"], item["id"])
            self.assertEqual(done["item"], item)
            self.assertEqual(added["item"]["status"], "in_progress")
            self.assertEqual(item["status"], "completed")
        self.assertEqual(events[-1]["response"]["status"], "completed")

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

    def test_namespaced_function_and_custom_calls(self):
        acc = self.make(tools=[NAMESPACE])
        value = "*** Begin Patch\n*** End Patch\n"
        acc.consume(
            packet([(
                TOOL,
                tool_text("workspace.read")
                + tool_text("workspace.patch", {"input": value}),
            )])
        )
        acc.finish(finish())
        self.assertEqual(
            [(item.type, item.name) for item in acc.response.output],
            [("function_call", "workspace.read"), ("custom_tool_call", "workspace.patch")],
        )
        self.assertEqual(acc.response.output[1].input, value)
        snapshot = acc.snapshot()
        self.assertEqual(snapshot["tools"][0]["type"], "namespace")
        self.assertEqual(snapshot["tools"][0]["tools"][0]["name"], "read")

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
        events = [
            json.loads(line[6:])
            for line in streamed.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(events[-1]["response"]["output"][0]["content"], result["output"][0]["content"])
        self.assertEqual(events[-1]["response"]["usage"], result["usage"])
        self.assertNotIn("[DONE]", streamed.text)

    async def test_validation_error_shape(self):
        response = await self.client.post("/v1/responses", json={"input": "x", "store": True})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "store")

    async def test_validation_error_names_the_offending_field(self):
        """A fault inside `input` must not surface as the union's string branch."""
        cases = [
            ({"input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": "https://a/b.png", "detail": "ultra"}]}]},
             "input[0].content[0].detail"),
            ({"input": [{"role": "critic", "content": "x"}]}, "input[0].role"),
            ({"input": [{"type": "function_call", "name": "f", "arguments": "{}"}]},
             "input[0].call_id"),
            ({"input": [{"type": "web_search_call", "id": "w"}]}, "input[0]"),
            ({"input": "x", "tools": [{"type": "namespace", "name": "ns",
                                       "tools": [{"type": "function"}]}]},
             "tools[0].tools[0].name"),
            # A real field whose name collides with a union tag must survive.
            ({"input": "x", "text": {"format": {"type": "bogus"}}}, "text.format.type"),
            ({"input": "x", "max_output_tokens": 15}, "max_output_tokens"),
            ({"input": "x", "metadata": {"key": "v" * 513}}, "metadata"),
        ]
        for payload, param in cases:
            response = await self.client.post("/v1/responses", json=payload)
            self.assertEqual(response.status_code, 400, response.text)
            self.assertEqual(response.json()["error"]["param"], param)

    async def test_unparsable_body_has_no_param(self):
        response = await self.client.post(
            "/v1/responses", content=b"{oops",
            headers={"content-type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIsNone(response.json()["error"]["param"])

    async def test_namespace_json_and_sse(self):
        self.container.chunks = [{"text": tool_text("workspace.read")}, finish()]
        payload = {"input": "use tool", "tools": [NAMESPACE]}
        response = await self.client.post("/v1/responses", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["tools"][0]["type"], "namespace")
        self.assertEqual(result["output"][0]["name"], "workspace.read")

        streamed = await self.client.post("/v1/responses", json={**payload, "stream": True})
        self.assertEqual(streamed.status_code, 200, streamed.text)
        events = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: ")]
        completed = events[-1]["response"]
        self.assertEqual(completed["tools"][0]["type"], "namespace")
        self.assertEqual(completed["output"][0]["name"], "workspace.read")

    async def test_hosted_tool_accepted_and_echoed(self):
        payload = {"input": "hi", "tools": [{"type": "web_search"}, FUNCTION]}
        response = await self.client.post("/v1/responses", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["tools"][0], {"type": "web_search"})
        self.assertEqual(result["tools"][1]["name"], "read")

        streamed = await self.client.post("/v1/responses", json={**payload, "stream": True})
        self.assertEqual(streamed.status_code, 200, streamed.text)
        events = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: ")]
        completed = events[-1]["response"]
        self.assertEqual(completed["tools"][0], {"type": "web_search"})

    async def test_openai_sdk_create_and_stream(self):
        import httpx2
        from openai import AsyncOpenAI

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=make_app()), base_url="http://test"
        ) as asgi_client:
            async def handler(request):
                response = await asgi_client.request(
                    request.method,
                    request.url.path,
                    headers=dict(request.headers),
                    content=request.content,
                )
                return httpx2.Response(
                    response.status_code,
                    headers=dict(response.headers),
                    content=response.content,
                    request=request,
                )

            transport = httpx2.MockTransport(handler)
            async with httpx2.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                sdk = AsyncOpenAI(
                    api_key="test",
                    base_url="http://test/v1",
                    http_client=client,
                    max_retries=0,
                )
                sdk._platform = "Linux"
                result = await sdk.responses.create(input="hi", model="test-model", store=False)
                self.assertEqual(result.output_text, "hello")
                stream = await sdk.responses.create(
                    input="hi", model="test-model", stream=True, store=False
                )
                events = [event async for event in stream]
                self.assertEqual(events[-1].type, "response.completed")
                self.assertEqual(events[-1].response.output_text, "hello")


class AdditionalContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_python_sdk_tool_round_trips(self):
        import httpx2
        from openai import AsyncOpenAI
        for tool in (FUNCTION, CUSTOM):
            value = "*** Begin Patch\n*** End Patch\n"
            name = tool["name"]
            args = {"path": "x"} if name == "read" else {"input": value}
            container = FakeContainer([{"text": tool_text(name, args)}, finish()])
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=make_app()), base_url="http://test"
            ) as asgi_client:
                async def handler(request):
                    response = await asgi_client.request(
                        request.method,
                        request.url.path,
                        headers=dict(request.headers),
                        content=request.content,
                    )
                    return httpx2.Response(
                        response.status_code,
                        headers=dict(response.headers),
                        content=response.content,
                        request=request,
                    )

                async with httpx2.AsyncClient(
                    transport=httpx2.MockTransport(handler), base_url="http://test"
                ) as client:
                    sdk = AsyncOpenAI(
                        api_key="test",
                        base_url="http://test/v1",
                        http_client=client,
                        max_retries=0,
                    )
                    sdk._platform = "Linux"
                    with patch.object(model, "container", container), patch.object(
                        model, "check_context_length"
                    ):
                        stream = await sdk.responses.create(
                            model="test-model", input="use tool", tools=[tool], stream=True
                        )
                        events = [e async for e in stream]
                        response = events[-1].response
                        self.assertEqual(response.status, "completed")
                        call = response.output[0]
                        expected = "function_call" if name == "read" else "custom_tool_call"
                        self.assertEqual(call.type, expected)
                        if name == "patch":
                            self.assertEqual(call.input, value)
                        container.chunks = [{"text": "done"}, finish()]
                        # SDK dumps contain optional nulls; exclude them when replaying.
                        input_items = [
                            {"role": "user", "content": "use tool"},
                            call.model_dump(exclude_none=True),
                            {
                                "type": call.type + "_output",
                                "call_id": call.call_id,
                                "output": "ok",
                            },
                        ]
                        result = await sdk.responses.create(
                            model="test-model", input=input_items, tools=[tool]
                        )
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
