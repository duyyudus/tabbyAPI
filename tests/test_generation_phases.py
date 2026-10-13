"""Reasoning/content phase switching and JSON lead stripping; no model weights or GPU required."""

import unittest
from unittest.mock import patch

import httpx

from common import model
from endpoints.OAI.types.responses import ResponsesRequest
from endpoints.OAI.utils.chat_completion import iter_generation_events
from endpoints.OAI.utils.responses_input import adapt_request
from endpoints.OAI.utils.stream_parser import CONTENT, REASONING, TOOL
from test_messages import TokenContainer, make_app as make_messages_app, sse_events
from test_responses import FakeContainer, finish, make_app as make_responses_app, tool_text


SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}

# A JSON answer after a reasoning block, with the blank line the grammar allows
# before it split across chunks
JSON_AFTER_REASONING = [
    {"text": "<think>hmm"},
    {"text": "</think>\n"},
    {"text": "\n "},
    {"text": '{"a"'},
    {"text": ": 1}\n"},
    finish(),
]


class PhaseMixin:
    """Records what the generator tells the backend about the phase."""

    def __init__(self, chunks=None, refusals=0):
        super().__init__(chunks)
        self.phases = []
        self.refusals = refusals
        self.stream_kwargs = None

    def stream_generate(self, *args, **kwargs):
        self.stream_kwargs = kwargs
        return super().stream_generate(*args, **kwargs)

    def set_generation_phase(self, request_id, reasoning):
        self.phases.append(reasoning)
        if self.refusals:
            self.refusals -= 1
            return False
        return True


class PhaseContainer(PhaseMixin, FakeContainer):
    pass


class PhaseTokenContainer(PhaseMixin, TokenContainer):
    pass


async def run(container, start_in_reasoning=False, **fields):
    params, _ = adapt_request(ResponsesRequest(input="x"))
    for name, value in fields.items():
        setattr(params, name, value)
    with patch.object(model, "container", container):
        source = iter_generation_events("id", "prompt", params, start_in_reasoning)
        return [p async for p in source]


def channel_text(packets, channel):
    return "".join(sub for p in packets for c, sub in p["events"] if c == channel)


class PhaseSwitchTests(unittest.IsolatedAsyncioTestCase):
    async def test_starting_phase_is_passed_to_the_backend(self):
        for start in (False, True):
            container = PhaseContainer([{"text": "hmm</think>hi"}, finish()])
            await run(container, start_in_reasoning=start)
            self.assertIs(container.stream_kwargs["reasoning_phase"], start)
            self.assertNotIn("filter_trigger", container.stream_kwargs)

    async def test_backend_is_told_of_each_transition(self):
        container = PhaseContainer(
            [{"text": "<think>hmm"}, {"text": " more"}, {"text": "</think>hi"}, finish()]
        )
        await run(container)
        self.assertEqual(container.phases, [True, False])

    async def test_no_switch_without_a_transition(self):
        container = PhaseContainer([{"text": "hello"}, {"text": " there"}, finish()])
        await run(container)
        self.assertEqual(container.phases, [])

    async def test_refused_switch_is_retried_on_the_next_chunk(self):
        container = PhaseContainer(
            [{"text": "<think>hmm</think>"}, {"text": "a"}, {"text": "b"}, {"text": "c"}, finish()],
            refusals=2,
        )
        # In and out of reasoning within one chunk: nothing to switch
        await run(container)
        self.assertEqual(container.phases, [])

        container = PhaseContainer(
            [{"text": "<think>a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}, finish()],
            refusals=2,
        )
        await run(container)
        self.assertEqual(container.phases, [True, True, True])

    async def test_no_switch_on_the_finishing_chunk(self):
        container = PhaseContainer([{"text": "<think>hmm", "finish_reason": "length"}])
        await run(container)
        self.assertEqual(container.phases, [])

    async def test_tool_call_inside_reasoning_stays_in_the_reasoning_phase(self):
        container = PhaseContainer(
            [{"text": "<think>hmm"}, {"text": tool_text()}, {"text": " ok"}, finish()]
        )
        packets = await run(container)
        self.assertTrue(channel_text(packets, TOOL))
        self.assertEqual(container.phases, [True])


class JsonLeadTests(unittest.IsolatedAsyncioTestCase):
    async def test_leading_whitespace_is_stripped_across_chunks(self):
        packets = await run(PhaseContainer(JSON_AFTER_REASONING), json_schema=SCHEMA)
        self.assertEqual(channel_text(packets, CONTENT), '{"a": 1}\n')
        self.assertEqual(channel_text(packets, REASONING), "hmm")

    async def test_whitespace_only_chunks_yield_no_content_event(self):
        packets = await run(PhaseContainer(JSON_AFTER_REASONING), json_schema=SCHEMA)
        for item in packets:
            for channel, sub in item["events"]:
                if channel == CONTENT:
                    self.assertTrue(sub.strip(), item)

    async def test_backend_text_is_left_unmodified(self):
        packets = await run(PhaseContainer(JSON_AFTER_REASONING), json_schema=SCHEMA)
        self.assertEqual(
            [p["generation"].get("text") for p in packets],
            [c.get("text") for c in JSON_AFTER_REASONING],
        )

    async def test_nothing_is_stripped_without_a_json_schema(self):
        packets = await run(PhaseContainer(JSON_AFTER_REASONING))
        self.assertEqual(channel_text(packets, CONTENT), '\n\n {"a": 1}\n')

        packets = await run(PhaseContainer(JSON_AFTER_REASONING), regex_pattern=".*")
        self.assertEqual(channel_text(packets, CONTENT), '\n\n {"a": 1}\n')


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def post(self, app, container, path, body):
        patches = [
            patch.object(model, "container", container),
            patch.object(model, "check_context_length"),
        ]
        for item in patches:
            item.start()
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.post(path, json=body)
        finally:
            for item in patches:
                item.stop()

    async def test_responses_json_format_after_reasoning(self):
        body = {
            "input": "hi",
            "text": {"format": {"type": "json_schema", "name": "out", "schema": SCHEMA}},
        }
        container = PhaseContainer(JSON_AFTER_REASONING)
        response = await self.post(make_responses_app(), container, "/v1/responses", body)
        self.assertEqual(response.status_code, 200, response.text)
        messages = [item for item in response.json()["output"] if item["type"] == "message"]
        self.assertEqual(len(messages), 1)
        self.assertEqual([part["text"] for part in messages[0]["content"]], ['{"a": 1}\n'])
        self.assertEqual(container.phases, [True, False])

        container = PhaseContainer(JSON_AFTER_REASONING)
        streamed = await self.post(
            make_responses_app(), container, "/v1/responses", body | {"stream": True}
        )
        events = sse_events(streamed.text)
        self.assertEqual(events[-1]["type"], "response.completed")
        text = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
        self.assertEqual(text, '{"a": 1}\n')

    async def test_messages_json_format_after_thinking(self):
        body = {
            "model": "test-model",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "adaptive"},
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
        }
        container = PhaseTokenContainer(JSON_AFTER_REASONING)
        response = await self.post(make_messages_app(), container, "/v1/messages", body)
        self.assertEqual(response.status_code, 200, response.text)
        content = response.json()["content"]
        self.assertEqual([block["type"] for block in content], ["thinking", "text"])
        self.assertEqual(content[1]["text"], '{"a": 1}\n')
        self.assertEqual(container.phases, [True, False])

        container = PhaseTokenContainer(JSON_AFTER_REASONING)
        streamed = await self.post(
            make_messages_app(), container, "/v1/messages", body | {"stream": True}
        )
        events = sse_events(streamed.text)
        self.assertEqual(events[-1]["type"], "message_stop")
        starts = [e["content_block"]["type"] for e in events if e["type"] == "content_block_start"]
        self.assertEqual(starts, ["thinking", "text"])
        text = "".join(
            e["delta"]["text"]
            for e in events
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
        )
        self.assertEqual(text, '{"a": 1}\n')
