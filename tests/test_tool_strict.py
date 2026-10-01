"""Strict tool-call validation for the Hermes, Kimi, OLMo 3 and LFM2 parsers."""

import json
import unittest

from endpoints.OAI.utils.tools import parse_toolcalls


def hermes_call(body):
    return f"<tool_call>\n{body}\n</tool_call>"


def kimi_section(*calls):
    inner = "".join(
        f"<|tool_call_begin|>functions.{name}:{i}<|tool_call_argument_begin|>{args}"
        "<|tool_call_end|>"
        for i, (name, args) in enumerate(calls)
    )
    return f"<|tool_calls_section_begin|>{inner}<|tool_calls_section_end|>"


def olmo3_block(*lines):
    return "<function_calls>\n" + "\n".join(lines) + "\n</function_calls>"


def lfm2_block(body):
    return f"<|tool_call_start|>{body}<|tool_call_end|>"


class StrictParseMixin:
    tool_format = None

    def parse(self, text):
        return parse_toolcalls(text, self.tool_format, strict=True)

    def assert_rejected(self, text):
        with self.assertRaises(ValueError):
            self.parse(text)
        # Non-strict parsing stays lenient and never raises
        parse_toolcalls(text, self.tool_format)

    def args(self, call):
        return json.loads(call.function.arguments)


class HermesStrictTests(StrictParseMixin, unittest.TestCase):
    tool_format = "hermes"

    def test_parallel_calls_accepted(self):
        text = hermes_call('{"name": "a", "arguments": {"x": 1}}') + hermes_call(
            '{"name": "b", "arguments": "{\\"y\\": 2}"}'
        )
        calls = self.parse(text)
        self.assertEqual([c.function.name for c in calls], ["a", "b"])
        self.assertEqual(self.args(calls[1]), {"y": 2})

    def test_null_arguments_accepted(self):
        calls = self.parse(hermes_call('{"name": "a", "arguments": null}'))
        self.assertEqual(self.args(calls[0]), {})

    def test_dropped_block_rejected(self):
        good = hermes_call('{"name": "a", "arguments": {}}')
        self.assert_rejected(good + hermes_call('{"name": "b", "arguments": {'))
        self.assert_rejected(good + hermes_call('{"arguments": {}}'))

    def test_non_json_string_arguments_rejected(self):
        self.assert_rejected(hermes_call('{"name": "a", "arguments": "not json"}'))

    def test_non_object_arguments_rejected(self):
        self.assert_rejected(hermes_call('{"name": "a", "arguments": [1, 2]}'))


class KimiStrictTests(StrictParseMixin, unittest.TestCase):
    tool_format = "kimi"

    def test_parallel_calls_accepted(self):
        calls = self.parse(kimi_section(("a", '{"x": 1}'), ("b", "")))
        self.assertEqual([c.function.name for c in calls], ["a", "b"])
        self.assertEqual(self.args(calls[1]), {})

    def test_dropped_call_rejected(self):
        broken = "<|tool_call_begin|>functions.b:1 {}<|tool_call_end|>"
        text = kimi_section(("a", "{}")).replace(
            "<|tool_calls_section_end|>", broken + "<|tool_calls_section_end|>"
        )
        self.assert_rejected(text)

    def test_invalid_json_arguments_rejected(self):
        self.assert_rejected(kimi_section(("a", '{"x": ')))

    def test_non_object_arguments_rejected(self):
        self.assert_rejected(kimi_section(("a", "null")))
        self.assert_rejected(kimi_section(("a", "[1]")))


class Olmo3StrictTests(StrictParseMixin, unittest.TestCase):
    tool_format = "olmo3"

    def test_parallel_and_multiline_calls_accepted(self):
        text = olmo3_block('a(x=1, y="s")', "b(z={", '  "k": [1, 2]', "})", "c()")
        calls = self.parse(text)
        self.assertEqual([c.function.name for c in calls], ["a", "b", "c"])
        self.assertEqual(self.args(calls[1]), {"z": {"k": [1, 2]}})

    def test_unparseable_line_rejected(self):
        self.assert_rejected(olmo3_block("a(x=1)", "not a call"))

    def test_positional_argument_rejected(self):
        self.assert_rejected(olmo3_block("a(1, x=2)"))

    def test_duplicate_argument_rejected(self):
        self.assert_rejected(olmo3_block("a(x=1, x=2)"))

    def test_unterminated_block_rejected(self):
        self.assert_rejected("<function_calls>\na(x=1)")


class Lfm2StrictTests(StrictParseMixin, unittest.TestCase):
    tool_format = "lfm2"

    def test_parallel_calls_accepted(self):
        calls = self.parse(lfm2_block("[a(x=1), b(from='y')]"))
        self.assertEqual([c.function.name for c in calls], ["a", "b"])
        self.assertEqual(self.args(calls[1]), {"from": "y"})

    def test_unparseable_block_rejected(self):
        self.assert_rejected(lfm2_block("[a(x=1)]") + lfm2_block("[b(x=]"))

    def test_positional_argument_rejected(self):
        self.assert_rejected(lfm2_block("[a(1, x=2)]"))
        self.assert_rejected(lfm2_block("[a(**kw)]"))

    def test_duplicate_argument_rejected(self):
        self.assert_rejected(lfm2_block("[a(x=1, x=2)]"))

    def test_non_literal_argument_rejected(self):
        self.assert_rejected(lfm2_block("[a(x=some_name)]"))

    def test_non_call_list_rejected(self):
        self.assert_rejected(lfm2_block("[a(x=1), 5]"))


if __name__ == "__main__":
    unittest.main()
