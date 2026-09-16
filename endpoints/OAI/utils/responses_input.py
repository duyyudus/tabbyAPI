"""Validate Responses capabilities and adapt items to model chat templates."""

import json
import re
from collections import Counter
from copy import deepcopy

from jsonschema import Draft202012Validator
from referencing import Registry
from lark import Lark

from common.logger import xlogger
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.types.responses import (
    AllowedTools,
    FunctionTool,
    InputMessage,
    NamedToolChoice,
    NamespaceTool,
    ReasoningItem,
    ToolResult,
    UnsupportedTool,
)


class ResponseRequestError(ValueError):
    def __init__(self, message, param=None, code="invalid_value"):
        super().__init__(message)
        self.param = param
        self.code = code


class InvalidModelOutput(ValueError):
    pass


def check_schema(schema):
    """Validate locally; remote references must never trigger network retrieval."""
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in {"$ref", "$dynamicRef"} and not str(value).startswith("#"):
                raise ResponseRequestError("Only local JSON schema references are supported")
            check_schema(value)
    elif isinstance(schema, list):
        for value in schema:
            check_schema(value)


def schema_validator(schema):
    try:
        check_schema(schema)
        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema, registry=Registry())
    except ResponseRequestError:
        raise
    except Exception as exc:
        raise ResponseRequestError(f"Invalid JSON schema: {exc}") from exc


def normalize_strict(schema, explicit=False):
    """Normalize supported object schemas, without changing property value types."""
    result = deepcopy(schema)

    def visit(node):
        if not isinstance(node, dict):
            return
        unsupported = {"format", "patternProperties", "unevaluatedProperties", "$dynamicRef"}
        if unsupported.intersection(node):
            raise ResponseRequestError(
                "Unsupported strict schema constraint: "
                + ", ".join(sorted(unsupported.intersection(node)))
            )
        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties", {})
            if node.get("additionalProperties") not in (None, False):
                raise ResponseRequestError("Strict schemas require additionalProperties:false")
            if explicit and (
                node.get("additionalProperties") is not False
                or set(node.get("required", [])) != set(properties)
            ):
                raise ResponseRequestError("Strict schemas must require every property")
            node["additionalProperties"] = False
            node["required"] = list(properties)
        for key in ("properties", "$defs", "definitions", "patternProperties"):
            for child in node.get(key, {}).values():
                visit(child)
        for key in ("items", "additionalProperties", "not", "if", "then", "else"):
            visit(node.get(key))
        for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
            for child in node.get(key, []):
                visit(child)

    visit(result)
    schema_validator(result)
    return result


# Fields that identify a hosted tool without exposing its configuration: an MCP
# definition can also carry server URLs and authorization headers.
HOSTED_TOOL_LABELS = ("name", "server_label")
# Tool sets already warned about. Agent clients resend the same tools every turn,
# so repeats drop to debug; the set is bounded because tool types are client-chosen.
_warned_tool_sets = set()


def loggable(value, limit=64):
    """Reduce client-supplied text to one printable console line."""
    text = " ".join("".join(c if c.isprintable() else " " for c in str(value)).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def describe_ignored_tool(definition):
    # One token per tool, e.g. mcp:github, so console wrapping only breaks between tools.
    kind = loggable(definition.get("type", "unknown"))
    label = next((definition[key] for key in HOSTED_TOOL_LABELS if definition.get(key)), None)
    return f"{kind}:{loggable(label).replace(' ', '_')}" if label is not None else kind


def warn_ignored_tools(definitions):
    counts = Counter(describe_ignored_tool(definition) for definition in definitions)
    listed = ", ".join(name if count == 1 else f"{name}(x{count})" for name, count in counts.items())
    message = "Ignoring hosted or unknown tool types, which are not executed locally:"
    types = ", ".join(sorted({str(definition.get("type", "unknown")) for definition in definitions}))
    extra = {"types": types, "tools": listed}
    signature = frozenset(counts.items())
    if signature in _warned_tool_sets:
        xlogger.debug(message, extra, details=listed)
        return
    if len(_warned_tool_sets) >= 256:
        _warned_tool_sets.clear()
    _warned_tool_sets.add(signature)
    xlogger.warning(message, extra, details=listed)


class ToolAdapter:
    def __init__(self, data):
        self.tools = {}
        self.validators = {}
        self.grammars = {}
        self.chat_tools = []
        self.ignored_types = []
        self.parallel = data.parallel_tool_calls
        self.required = data.tool_choice == "required"
        ignored = []
        for tool in data.tools:
            if isinstance(tool, UnsupportedTool):
                self.ignored_types.append(tool.definition.get("type", "unknown"))
                ignored.append(tool.definition)
            elif isinstance(tool, NamespaceTool):
                for child in tool.tools:
                    self._register_tool(child, f"{tool.name}.{child.name}", tool.description)
            else:
                self._register_tool(tool, tool.name)
        if ignored:
            warn_ignored_tools(ignored)
        self.allowed = set(self.tools)
        choice = data.tool_choice
        if choice == "none":
            self.allowed = set()
        elif isinstance(choice, NamedToolChoice):
            self.allowed = {self._choice_name(choice)}
            self.required = True
        elif isinstance(choice, AllowedTools):
            self.allowed = {self._choice_name(item) for item in choice.tools}
            self.required = choice.mode == "required"
        if self.required and not self.allowed:
            raise ResponseRequestError("tool_choice requires at least one tool", "tool_choice")
        self.chat_tools = [t for t in self.chat_tools if t["function"]["name"] in self.allowed]

    def _register_tool(self, tool, name, namespace_description=""):
        if name in self.tools:
            raise ResponseRequestError("Tool names must be unique", "tools")
        self.tools[name] = tool
        description = "\n".join(part for part in (namespace_description, tool.description) if part)
        if isinstance(tool, FunctionTool):
            schema_validator(tool.parameters)
            if tool.parameters.get("type") != "object":
                raise ResponseRequestError("Function parameters must be an object schema", "tools")
            if tool.strict is not False:
                try:
                    tool.parameters = normalize_strict(tool.parameters, tool.strict is True)
                    tool.strict = True
                except ResponseRequestError:
                    if tool.strict is True:
                        raise
                    tool.strict = False
            parameters = tool.parameters
            if tool.strict:
                self.validators[name] = schema_validator(parameters)
        else:
            parameters = {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": False,
            }
            self.validators[name] = schema_validator(parameters)
            description += ("\n" if description else "") + (
                "Pass the exact raw tool input in the input string."
            )
            fmt = tool.format
            if fmt.type == "grammar":
                if fmt.syntax != "lark" or not fmt.definition:
                    raise ResponseRequestError("Custom grammars require a Lark definition", "tools")
                # Lark allows local file imports; only bundled common terminals are allowed.
                for line in fmt.definition.splitlines():
                    if "%import" in line and not re.fullmatch(
                        r"\s*%import common\.[A-Za-z_][A-Za-z_0-9]*(?:\s*->\s*\w+)?\s*", line
                    ):
                        raise ResponseRequestError(
                            "Only common terminal imports are supported", "tools"
                        )
                try:
                    self.grammars[name] = Lark(fmt.definition, parser="earley")
                except Exception as exc:
                    raise ResponseRequestError(f"Unsupported Lark grammar: {exc}", "tools") from exc
                description += "\nThe input must match this Lark grammar:\n" + fmt.definition
            elif fmt.syntax is not None or fmt.definition is not None:
                raise ResponseRequestError("Text tools cannot specify a grammar", "tools")
        self.chat_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )

    def _choice_name(self, choice):
        tool = self.tools.get(choice.name)
        if tool is None or tool.type != choice.type:
            raise ResponseRequestError("tool_choice must reference a declared tool", "tool_choice")
        return choice.name

    def validate_calls(self, calls):
        if self.required and not calls:
            raise InvalidModelOutput("The model did not produce the required tool call")
        if len({call.id for call in calls}) != len(calls):
            raise InvalidModelOutput("The model generated duplicate tool call IDs")
        if not self.parallel and len(calls) > 1:
            raise InvalidModelOutput(
                "The model generated multiple calls with parallel_tool_calls:false"
            )
        for call in calls:
            name = call.function.name
            if name not in self.allowed:
                raise InvalidModelOutput(f"The model called an unavailable tool: {name}")
            try:
                args = json.loads(call.function.arguments)
                if not isinstance(args, dict):
                    raise ValueError("Function arguments must be a JSON object")
                if name in self.validators:
                    self.validators[name].validate(args)
                if name in self.grammars:
                    self.grammars[name].parse(args["input"])
            except Exception as exc:
                raise InvalidModelOutput(f"Invalid arguments for tool {name}: {exc}") from exc
        return calls


def convert_content(content, vision):
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if part.type in {"input_text", "output_text"}:
            parts.append({"type": "text", "text": part.text})
        else:
            if not vision:
                raise ResponseRequestError("The loaded model does not support images", "input")
            if not part.image_url.startswith(("https://", "http://", "data:image/")):
                raise ResponseRequestError("Images require HTTP(S) or image data URLs", "input")
            parts.append({"type": "image_url", "image_url": {"url": part.image_url}})
    return parts


def adapt_request(data, vision=False):
    tools = ToolAdapter(data)
    messages = []
    if data.instructions is not None:
        messages.append({"role": "system", "content": data.instructions})
    items = (
        [InputMessage(role="user", content=data.input)]
        if isinstance(data.input, str)
        else data.input
    )
    calls = {}
    answered = set()
    for item in items:
        if isinstance(item, ReasoningItem):
            # Replayed reasoning has no representation in the chat context; drop it
            # without disturbing call/result pairing around it.
            continue
        elif isinstance(item, InputMessage):
            messages.append({"role": item.role, "content": convert_content(item.content, vision)})
        elif isinstance(item, ToolResult):
            if item.call_id not in calls or item.call_id in answered:
                raise ResponseRequestError("Tool output requires a unique preceding call", "input")
            expected = calls[item.call_id] + "_output"
            if item.type != expected:
                raise ResponseRequestError("Tool output type does not match its call", "input")
            answered.add(item.call_id)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.call_id,
                    "content": convert_content(item.output, vision),
                }
            )
        else:
            if item.call_id in calls:
                raise ResponseRequestError("Duplicate call_id", "input")
            calls[item.call_id] = item.type
            arguments = (
                item.arguments
                if item.type == "function_call"
                else json.dumps({"input": item.input})
            )
            try:
                if not isinstance(json.loads(arguments), dict):
                    raise ValueError()
            except ValueError as exc:
                raise ResponseRequestError(
                    "Replayed function arguments must be a JSON object", "input"
                ) from exc
            call = {
                "id": item.call_id,
                "type": "function",
                "function": {"name": item.name, "arguments": arguments},
            }
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "tool_calls": [call]})
    if set(calls) != answered:
        raise ResponseRequestError(
            "Every replayed call needs a tool result before generation", "input"
        )
    if not messages:
        raise ResponseRequestError("Provide input or instructions", "input")
    schema = None
    fmt = data.text.format
    if fmt.type != "json_schema" and any(
        v is not None for v in (fmt.schema_, fmt.name, fmt.strict, fmt.description)
    ):
        raise ResponseRequestError("Schema options require json_schema format", "text.format")
    if fmt.type == "json_object":
        schema = {"type": "object"}
    elif fmt.type == "json_schema":
        if fmt.schema_ is None or not fmt.name:
            raise ResponseRequestError("json_schema requires name and schema", "text.format")
        schema = fmt.schema_
        if fmt.strict:
            schema = normalize_strict(schema, explicit=True)
        schema_validator(schema)
    if schema is not None and not schema:
        schema = {"allOf": [{}]}
    if schema is not None and tools.chat_tools:
        raise ResponseRequestError(
            "Combining structured text and tools is unsupported", "text.format"
        )
    params = {
        k: getattr(data, k)
        for k in ("temperature", "top_p", "model")
        if getattr(data, k) is not None
    }
    if data.max_output_tokens is not None:
        params["max_tokens"] = data.max_output_tokens
    # Always parse tool tags, including when none are allowed, to detect violations.
    chat = ChatCompletionRequest(
        messages=messages,
        tools=tools.chat_tools,
        n=1,
        tool_choice="auto",
        json_schema=schema,
        reasoning_effort=data.reasoning.effort if data.reasoning else None,
        verbosity=data.text.verbosity,
        **params,
    )
    chat._fail_on_grammar_error = True
    if tools.required:
        names = ", ".join(sorted(tools.allowed))
        chat.messages.insert(
            0,
            type(chat.messages[0])(
                role="system", content=f"You must call one of these tools this turn: {names}."
            ),
        )
    return chat, tools
