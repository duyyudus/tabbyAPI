"""Anthropic Messages contract tests; no model weights or GPU required."""

import json
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from pydantic import ValidationError

from common import model
from common.auth import check_api_key
from common.errors import ContextLengthHTTPException
from common.networking import add_request_id
from endpoints.Anthropic.router import router
from endpoints.Anthropic.types.messages import CountTokensRequest, MessagesRequest
from endpoints.Anthropic.utils.messages import MessagesAccumulator, collect_message
from endpoints.Anthropic.utils.messages_input import MessagesRequestError, adapt_request
from endpoints.OAI.utils.responses_input import InvalidModelOutput
from endpoints.OAI.utils.stream_parser import CONTENT, REASONING, TOOL
from test_responses import FakeContainer, finish, packet, tool_text

READ = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}


def request(**fields):
    return MessagesRequest(
        **{"model": "test-model", "max_tokens": 256, "messages": [{"role": "user", "content": "hi"}]}
        | fields
    )


class TokenContainer(FakeContainer):
    def encode_tokens(self, text, **kwargs):
        return list(range(len(text.split())))


def make_app():
    app = FastAPI()
    app.include_router(router)

    async def no_auth():
        pass

    app.dependency_overrides[check_api_key] = no_auth

    @app.middleware("http")
    async def request_id(request, call_next):
        await add_request_id(request)
        return await call_next(request)

    return app


def sse_events(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


class InputTests(unittest.TestCase):
    def test_system_blocks_drop_attribution_header(self):
        data = request(system=[
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=abc;"},
            {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Be brief."},
        ])
        chat, _ = adapt_request(data)
        self.assertEqual(chat.messages[0].role, "system")
        self.assertEqual(chat.messages[0].content, "You are helpful.\n\nBe brief.")

    def test_tool_round_trip_and_reasoning_replay(self):
        data = request(tools=[READ], messages=[
            {"role": "user", "content": "read x"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "I should read.", "signature": "sig"},
                {"type": "text", "text": "Reading."},
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "x"}},
                {"type": "tool_use", "id": "toolu_2", "name": "Read", "input": {"path": "y"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "Y"},
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": [{"type": "text", "text": "X"}], "is_error": False},
                {"type": "text", "text": "<system-reminder>ok</system-reminder>"},
            ]},
        ])
        chat, tools = adapt_request(data)
        roles = [m.role for m in chat.messages]
        self.assertEqual(roles, ["user", "assistant", "tool", "tool", "user"])
        assistant = chat.messages[1]
        self.assertEqual(assistant.reasoning_content, "I should read.")
        self.assertEqual(assistant.content, "Reading.")
        self.assertEqual([c.id for c in assistant.tool_calls], ["toolu_1", "toolu_2"])
        self.assertEqual(json.loads(assistant.tool_calls[0].function.arguments), {"path": "x"})
        self.assertEqual(chat.messages[3].content, "X")
        self.assertEqual(chat.tools[0].function.parameters, READ["input_schema"])
        self.assertTrue(tools.parallel)

    def test_tool_pairing_is_enforced(self):
        use = {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}]}
        for messages in (
            [{"role": "user", "content": "x"}, use, {"role": "user", "content": "no result"}],
            [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_9", "content": "x"}]}],
            [{"role": "user", "content": "x"}, use],
        ):
            with self.subTest(messages=messages), self.assertRaises(MessagesRequestError):
                adapt_request(request(tools=[READ], messages=messages))

    def test_consecutive_turns_merge_and_mid_conversation_system(self):
        data = request(messages=[
            {"role": "user", "content": "a"},
            {"role": "system", "content": "note"},
            {"role": "user", "content": [{"type": "text", "text": "b"}]},
        ])
        chat, _ = adapt_request(data)
        self.assertEqual([m.role for m in chat.messages], ["user"])
        self.assertEqual(chat.messages[0].content, "a\n\nnote\n\nb")

    def test_leading_system_message_joins_system_prompt(self):
        data = request(system="s", messages=[
            {"role": "system", "content": "t"}, {"role": "user", "content": "u"}])
        chat, _ = adapt_request(data)
        self.assertEqual(chat.messages[0].content, "s\n\nt")

    def test_prefill_continues_final_assistant_message(self):
        data = request(messages=[
            {"role": "user", "content": "x"}, {"role": "assistant", "content": "The answer is"}])
        chat, _ = adapt_request(data)
        self.assertTrue(chat.continue_final_message)
        self.assertFalse(chat.add_generation_prompt)
        with self.assertRaises(MessagesRequestError):
            adapt_request(request(messages=[
                {"role": "user", "content": "x"}, {"role": "assistant", "content": "trailing "}]))

    def test_thinking_mapping(self):
        chat, _ = adapt_request(request())
        self.assertFalse(chat.enable_thinking)
        chat, _ = adapt_request(request(thinking={"type": "adaptive"},
                                        output_config={"effort": "max"}))
        self.assertTrue(chat.enable_thinking)
        self.assertEqual(chat.reasoning_effort, "high")
        chat, _ = adapt_request(request(max_tokens=4096,
                                        thinking={"type": "enabled", "budget_tokens": 2048}))
        self.assertEqual(chat.reasoning_budget_tokens, 2048)
        with self.assertRaises(ValidationError):
            request(max_tokens=2048, thinking={"type": "enabled", "budget_tokens": 2048})
        with self.assertRaises(MessagesRequestError):
            adapt_request(request(tools=[READ], tool_choice={"type": "any"},
                                  thinking={"type": "adaptive"}))

    def test_sampling_and_stop_sequences(self):
        chat, _ = adapt_request(request(temperature=0.5, top_p=0.9, top_k=20,
                                        stop_sequences=["END"]))
        self.assertEqual((chat.temperature, chat.top_p, chat.top_k), (0.5, 0.9, 20))
        self.assertEqual(chat.stop, ["END"])
        self.assertEqual(chat.max_tokens, 256)

    def test_tool_choice(self):
        chat, tools = adapt_request(request(tools=[READ, {**READ, "name": "Write"}],
                                            tool_choice={"type": "tool", "name": "Write",
                                                         "disable_parallel_tool_use": True}))
        self.assertEqual([t.function.name for t in chat.tools], ["Write"])
        self.assertIn("Write", chat.messages[0].content)
        self.assertFalse(tools.parallel)
        chat, _ = adapt_request(request(tools=[READ], tool_choice={"type": "none"}))
        self.assertIsNone(chat.tools)
        with self.assertRaises(MessagesRequestError):
            adapt_request(request(tools=[READ], tool_choice={"type": "tool", "name": "Nope"}))

    def test_open_lists_are_accepted(self):
        """Claude Code sends fields and tools that have no local meaning."""
        data = request(
            metadata={"user_id": "u"},
            context_management={"edits": []},
            tools=[{**READ, "cache_control": {"type": "ephemeral"}, "defer_loading": False,
                    "eager_input_streaming": True},
                   {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                {"type": "tool_reference", "tool_name": "Read"},
            ]}],
        )
        chat, _ = adapt_request(data)
        self.assertEqual([t.function.name for t in chat.tools], ["Read"])
        self.assertEqual(chat.messages[-1].content, "hi")

    def test_rejections(self):
        for fields in (
            {"max_tokens": 0},
            {"messages": []},
            {"messages": [{"role": "tool", "content": "x"}]},
            {"tools": [{"name": "bad name", "input_schema": {"type": "object"}}]},
            {"temperature": 1.5},
            {"thinking": {"type": "enabled", "budget_tokens": 100}},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                request(**fields)
        for fields in (
            {"messages": [{"role": "user", "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                                "data": "AA=="}}]}]},
            {"messages": [{"role": "user", "content": [{"type": "mystery"}]}]},
            {"messages": [{"role": "user", "content": [{"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": "AA=="}}]}]},
            {"tools": [{"name": "t", "input_schema": {"type": "string"}}]},
            {"tools": [READ, READ]},
            {"tools": [READ], "output_config": {"format": {"type": "json_schema",
                                                           "schema": {"type": "object"}}}},
        ):
            with self.subTest(fields=fields), self.assertRaises(MessagesRequestError):
                adapt_request(request(**fields))

    def test_images_and_documents(self):
        chat, _ = adapt_request(request(messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
            {"type": "document", "title": "Doc", "source": {"type": "text", "data": "body"}},
            {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
        ]}]), vision=True)
        parts = chat.messages[0].content
        self.assertEqual(parts[0].image_url.url, "data:image/png;base64,AA==")
        self.assertEqual(parts[2].text, "Doc\n\nbody")
        self.assertEqual(parts[4].image_url.url, "https://x/y.png")

    def test_count_tokens_request_has_no_generation_fields(self):
        data = CountTokensRequest(model="m", messages=[{"role": "user", "content": "hi"}])
        chat, _ = adapt_request(data)
        self.assertEqual(chat.messages[0].content, "hi")


class AccumulatorTests(unittest.TestCase):
    def setUp(self):
        self.data = request(tools=[READ], thinking={"type": "adaptive"})
        _, self.tools = adapt_request(self.data)

    def run_packets(self, accumulator, packets, generation):
        events = accumulator.start()
        for item in packets:
            events += accumulator.consume(item)
        return events + accumulator.finish(generation)

    def test_thinking_text_and_tool_stream(self):
        accumulator = MessagesAccumulator(self.data, self.tools, "test-model", 10)
        events = self.run_packets(accumulator, [
            packet([(REASONING, "hmm")]),
            packet([(CONTENT, "\n\n"), (CONTENT, "Let me read.")]),
            packet([(TOOL, tool_text("Read", {"path": "x"}))]),
            packet([(CONTENT, "\n")]),
        ], finish())
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds, [
            "message_start",
            "content_block_start", "content_block_delta",
            "content_block_delta", "content_block_stop",
            "content_block_start", "content_block_delta", "content_block_stop",
            "content_block_start", "content_block_delta", "content_block_stop",
            "message_delta", "message_stop",
        ])
        self.assertEqual(events[3]["delta"]["type"], "signature_delta")
        self.assertEqual(events[6]["delta"]["text"], "\n\nLet me read.")
        self.assertEqual(events[8]["content_block"]["type"], "tool_use")
        self.assertTrue(events[8]["content_block"]["id"].startswith("toolu_"))
        self.assertEqual(json.loads(events[9]["delta"]["partial_json"]), {"path": "x"})
        self.assertEqual([e.get("index") for e in events[1:11] if "index" in e],
                         [0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual(events[-2]["delta"], {"stop_reason": "tool_use", "stop_sequence": None})
        self.assertEqual(events[-2]["usage"], {
            "input_tokens": 7, "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 3, "output_tokens": 5})
        self.assertEqual(events[0]["message"]["usage"]["input_tokens"], 10)

    def test_thinking_hidden_unless_requested(self):
        data = request()
        _, tools = adapt_request(data)
        events = self.run_packets(MessagesAccumulator(data, tools, "m", 1),
                                  [packet([(REASONING, "secret"), (CONTENT, "hi")])], finish())
        self.assertEqual([e["content_block"]["type"] for e in events
                          if e["type"] == "content_block_start"], ["text"])

    def test_omitted_thinking_display(self):
        data = request(thinking={"type": "adaptive", "display": "omitted"})
        events = self.run_packets(MessagesAccumulator(data, self.tools, "m", 1),
                                  [packet([(REASONING, "secret")])], finish())
        self.assertFalse(any(e.get("delta", {}).get("type") == "thinking_delta" for e in events))
        self.assertTrue(any(e.get("delta", {}).get("type") == "signature_delta" for e in events))

    def test_stop_reasons(self):
        cases = [
            (finish(finish_reason="length", eos_reason="max_new_tokens"), "max_tokens", None),
            (finish(eos_reason="stop_string", stop_str="END"), "stop_sequence", "END"),
            (finish(eos_reason="stop_string", stop_str="<|im_end|>"), "end_turn", None),
            (finish(eos_reason="stop_token"), "end_turn", None),
        ]
        data = request(stop_sequences=["END"])
        for generation, reason, sequence in cases:
            with self.subTest(reason=reason):
                events = self.run_packets(MessagesAccumulator(data, self.tools, "m", 1),
                                          [packet([(CONTENT, "x")])], generation)
                self.assertEqual(events[-2]["delta"],
                                 {"stop_reason": reason, "stop_sequence": sequence})

    def test_limit_never_releases_tool(self):
        events = self.run_packets(
            MessagesAccumulator(self.data, self.tools, "m", 1),
            [packet([(TOOL, "<tool_call><function=Read><parameter=path>")])],
            finish(finish_reason="length"),
        )
        self.assertFalse(any(e["type"] == "content_block_start" for e in events))
        self.assertEqual(events[-2]["delta"]["stop_reason"], "max_tokens")

    def test_malformed_tool_call_fails(self):
        accumulator = MessagesAccumulator(self.data, self.tools, "m", 1)
        accumulator.consume(packet([(TOOL, "<tool_call><function=Read></tool_call>")]))
        with self.assertRaises(InvalidModelOutput):
            accumulator.finish(finish())

    def test_disabled_parallel_tool_use_keeps_first_call(self):
        data = request(tools=[READ], tool_choice={"type": "auto", "disable_parallel_tool_use": True})
        _, tools = adapt_request(data)
        events = self.run_packets(MessagesAccumulator(data, tools, "m", 1), [
            packet([(TOOL, tool_text("Read", {"path": "a"}) + tool_text("Read", {"path": "b"}))]),
        ], finish())
        starts = [e for e in events if e["type"] == "content_block_start"]
        self.assertEqual(len(starts), 1)

    def test_strict_tool_schema_enforced(self):
        data = request(tools=[{**READ, "strict": True}])
        _, tools = adapt_request(data)
        accumulator = MessagesAccumulator(data, tools, "m", 1)
        accumulator.consume(packet([(TOOL, tool_text("Read", {"path": 3}))]))
        with self.assertRaises(InvalidModelOutput):
            accumulator.finish(finish())


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.container = TokenContainer()
        self.patches = [
            patch.object(model, "container", self.container),
            patch.object(model, "check_context_length"),
        ]
        for item in self.patches:
            item.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=make_app()), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        for item in self.patches:
            item.stop()

    def body(self, **fields):
        return {"model": "test-model", "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}]} | fields

    async def test_json_and_sse_agree(self):
        self.container.chunks = [{"text": "<think>hmm</think>hello"}, finish()]
        body = self.body(thinking={"type": "adaptive"})
        response = await self.client.post("/v1/messages?beta=true", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        message = response.json()
        self.assertEqual(message["type"], "message")
        self.assertTrue(message["id"].startswith("msg_"))
        self.assertEqual(message["model"], "test-model")
        self.assertEqual([b["type"] for b in message["content"]], ["thinking", "text"])
        self.assertEqual(message["content"][1]["text"], "hello")
        self.assertEqual(message["stop_reason"], "end_turn")

        streamed = await self.client.post("/v1/messages", json=body | {"stream": True})
        self.assertIn("event: message_start", streamed.text)
        self.assertIn("event: message_stop", streamed.text)
        self.assertTrue(streamed.headers["content-type"].startswith("text/event-stream"))

        async def replay():
            for event in sse_events(streamed.text):
                yield event

        folded = await collect_message(replay())
        self.assertEqual(folded["content"][1], message["content"][1])
        self.assertEqual(folded["content"][0]["thinking"], "hmm")
        self.assertEqual(folded["usage"], message["usage"])

    async def test_tool_use_response(self):
        self.container.chunks = [{"text": tool_text("Read", {"path": "x"})}, finish()]
        response = await self.client.post("/v1/messages", json=self.body(tools=[READ]))
        message = response.json()
        self.assertEqual(message["stop_reason"], "tool_use")
        self.assertEqual(message["content"][0]["input"], {"path": "x"})
        self.assertEqual(message["content"][0]["name"], "Read")

    async def test_error_shapes(self):
        response = await self.client.post("/v1/messages", json={"model": "m", "messages": []})
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "invalid_request_error")

        response = await self.client.post("/v1/messages", json=self.body(
            messages=[{"role": "user", "content": [{"type": "text"}]}]))
        self.assertEqual(response.json()["error"]["message"],
                         "messages.0.content.0.text: Field required")

        response = await self.client.post("/v1/messages", content=b"{not json")
        self.assertEqual(response.status_code, 400)

        with patch.object(model, "check_context_length",
                          side_effect=ContextLengthHTTPException("Prompt length 9 exceeds 8")):
            response = await self.client.post("/v1/messages", json=self.body())
        self.assertEqual(response.status_code, 400)
        self.assertTrue(response.json()["error"]["message"].startswith("prompt is too long"))

    async def test_auth_error_shape(self):
        app = make_app()
        app.dependency_overrides.clear()
        with patch("common.auth.DISABLE_AUTH", False), patch("common.auth.AUTH_KEYS", None):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://test") as client:
                response = await client.post("/v1/messages", json=self.body())
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["type"], "authentication_error")

    async def test_generation_failure(self):
        self.container.chunks = [{"text": "partial"}, RuntimeError("boom")]
        response = await self.client.post("/v1/messages", json=self.body())
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["type"], "api_error")
        self.container.chunks = [{"text": "partial"}, RuntimeError("boom")]
        streamed = await self.client.post("/v1/messages", json=self.body(stream=True))
        events = sse_events(streamed.text)
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("event: error", streamed.text)
        self.assertTrue(self.container.closed)

    async def test_count_tokens(self):
        response = await self.client.post("/v1/messages/count_tokens", json={
            "model": "test-model", "messages": [{"role": "user", "content": "one two three"}]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreater(response.json()["input_tokens"], 0)

    async def test_anthropic_sdk(self):
        try:
            import anthropic
        except ImportError:
            self.skipTest("anthropic SDK not installed")
        # Newer SDK releases ship on the httpx2 fork and refuse httpx clients.
        try:
            import httpx2 as sdk_httpx
        except ImportError:
            sdk_httpx = httpx
        client = anthropic.AsyncAnthropic(
            api_key="x", base_url="http://test",
            http_client=sdk_httpx.AsyncClient(transport=sdk_httpx.ASGITransport(app=make_app())),
        )
        self.container.chunks = [{"text": "<think>plan</think>Reading."
                                  + tool_text("Read", {"path": "x"})}, finish()]
        message = await client.messages.create(
            model="test-model", max_tokens=64, tools=[READ],
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": "read x"}],
        )
        self.assertEqual([b.type for b in message.content], ["thinking", "text", "tool_use"])
        self.assertEqual(message.content[2].input, {"path": "x"})

        self.container.chunks = [{"text": "<think>plan</think>Reading."
                                  + tool_text("Read", {"path": "x"})}, finish()]
        async with client.messages.stream(
            model="test-model", max_tokens=64, tools=[READ], thinking={"type": "adaptive"},
            messages=[
                {"role": "user", "content": "read x"},
                {"role": "assistant", "content": [c.model_dump() for c in message.content]},
                {"role": "user", "content": [{"type": "tool_result",
                                              "tool_use_id": message.content[2].id,
                                              "content": "file"}]},
            ],
        ) as stream:
            text = "".join([chunk async for chunk in stream.text_stream])
            final = await stream.get_final_message()
        self.assertEqual(text, "Reading.")
        self.assertEqual(final.stop_reason, "tool_use")
        self.assertEqual(final.content[0].thinking, "plan")
        self.assertEqual(final.content[2].input, {"path": "x"})
        self.assertEqual(final.usage.cache_read_input_tokens, 3)

        count = await client.messages.count_tokens(
            model="test-model", messages=[{"role": "user", "content": "a b"}])
        self.assertGreater(count.input_tokens, 0)

        self.container.chunks = [finish()]
        with self.assertRaises(anthropic.BadRequestError):
            await client.messages.create(model="test-model", max_tokens=64, messages=[])
