"""Validate Messages requests and adapt them to the chat template pipeline."""

import json

from common.logger import xlogger
from endpoints.Anthropic.types.messages import (
    ContentBlockSource,
    DocumentBlock,
    ImageBlock,
    PlainTextSource,
    RedactedThinkingBlock,
    SearchResultBlock,
    ServerTool,
    TextBlock,
    ThinkingAdaptive,
    ThinkingEnabled,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.utils.responses_input import (
    ResponseRequestError,
    schema_validator,
    warn_ignored_tools,
)

# Claude Code prepends this block to `system`; the first-party API strips it when it
# arrives as the first system block, so it never reaches the model or its prompt cache.
ATTRIBUTION_PREFIX = "x-anthropic-billing-header:"

# Blocks produced by Anthropic-hosted tools. None run here, so replayed ones carry no
# context the model can act on and are dropped.
DROPPED_BLOCKS = {"server_tool_use", "tool_reference", "container_upload"}

# Templates commonly know low/medium/high only; stronger settings map to the highest.
EFFORTS = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}

BLOCK_SEPARATOR = "\n\n"


class MessagesRequestError(ValueError):
    """A client error, reported as invalid_request_error."""


def _schema_validator(schema):
    try:
        return schema_validator(schema)
    except ResponseRequestError as exc:
        raise MessagesRequestError(str(exc)) from exc


def _droppable(block):
    return block.type in DROPPED_BLOCKS or block.type.endswith("_tool_result")


class ToolSet:
    def __init__(self, data):
        self.tools = {}
        self.validators = {}
        self.chat_tools = []
        ignored = []
        for tool in data.tools or []:
            if isinstance(tool, ServerTool):
                ignored.append(tool.model_dump())
                continue
            if tool.name in self.tools:
                raise MessagesRequestError(f"tools: Tool names must be unique: {tool.name}")
            if tool.input_schema.get("type") != "object":
                raise MessagesRequestError(
                    f"tools: input_schema of tool {tool.name} must have type object"
                )
            self.tools[tool.name] = tool
            if tool.strict:
                self.validators[tool.name] = _schema_validator(tool.input_schema)
            self.chat_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
            )
        if ignored:
            warn_ignored_tools(ignored)

        choice = data.tool_choice
        self.parallel = not (choice and choice.disable_parallel_tool_use)
        self.required = choice is not None and choice.type in ("any", "tool")
        self.allowed = set(self.tools)
        if choice is not None and choice.type == "none":
            self.allowed = set()
        elif choice is not None and choice.type == "tool":
            if choice.name not in self.tools:
                raise MessagesRequestError(
                    f"tool_choice: Tool '{choice.name}' not found in the provided tools"
                )
            self.allowed = {choice.name}
        if self.required and not self.allowed:
            raise MessagesRequestError("tool_choice: requires at least one tool")
        self.chat_tools = [t for t in self.chat_tools if t["function"]["name"] in self.allowed]

    def validate_calls(self, calls):
        """Raise if a parsed call cannot be released to the client."""

        for call in calls:
            name = call.function.name
            arguments = json.loads(call.function.arguments)
            if not isinstance(arguments, dict):
                raise ValueError(f"Arguments for tool {name} are not a JSON object")
            if name not in self.allowed:
                # The client answers an unknown tool with an error result, which lets
                # the model recover instead of failing the whole turn.
                xlogger.warning(f"The model called a tool that is not available: {name}")
            if name in self.validators:
                self.validators[name].validate(arguments)


def system_text(system):
    if system is None or isinstance(system, str):
        return system or None
    blocks = list(system)
    if blocks and blocks[0].text.startswith(ATTRIBUTION_PREFIX):
        blocks = blocks[1:]
    return BLOCK_SEPARATOR.join(block.text for block in blocks) or None


def image_part(block, vision):
    if not vision:
        raise MessagesRequestError("The loaded model does not support image input")
    source = block.source
    if source.type == "base64":
        url = f"data:{source.media_type};base64,{source.data}"
    elif source.type == "url":
        url = source.url
    else:
        raise MessagesRequestError("File sources are not supported for images")
    return {"type": "image_url", "image_url": {"url": url}}


def content_parts(blocks, vision, where):
    """Convert user-side blocks to chat parts; the text of each block stays separate."""

    parts = []
    for block in blocks:
        if isinstance(block, TextBlock):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            parts.append(image_part(block, vision))
        elif isinstance(block, DocumentBlock):
            header = [text for text in (block.title, block.context) if text]
            source = block.source
            if isinstance(source, PlainTextSource):
                parts.append({"type": "text", "text": BLOCK_SEPARATOR.join(header + [source.data])})
            elif isinstance(source, ContentBlockSource):
                if header:
                    parts.append({"type": "text", "text": BLOCK_SEPARATOR.join(header)})
                inner = source.content
                inner = [TextBlock(text=inner)] if isinstance(inner, str) else inner
                parts += content_parts(inner, vision, where)
            else:
                raise MessagesRequestError(
                    f"{where}: Only plain text and content documents are supported"
                )
        elif isinstance(block, SearchResultBlock):
            text = BLOCK_SEPARATOR.join(item.text for item in block.content)
            parts.append(
                {"type": "text", "text": BLOCK_SEPARATOR.join((block.title, block.source, text))}
            )
        elif _droppable(block):
            xlogger.debug(f"Dropping replayed {block.type} block")
        else:
            raise MessagesRequestError(
                f"{where}: Unsupported content block type in this position: {block.type}"
            )
    return parts


def join_parts(parts):
    """Separate blocks, then collapse text-only content to the string templates expect."""

    joined = []
    for part in parts:
        if joined:
            joined.append({"type": "text", "text": BLOCK_SEPARATOR})
        joined.append(part)
    if all(part["type"] == "text" for part in joined):
        return "".join(part["text"] for part in joined)
    return joined


class Conversation:
    """Chat messages built in order; consecutive turns of the same role are merged."""

    def __init__(self, system):
        self.messages = []
        self.system = [system] if system else []

    def user(self, parts):
        if not parts:
            return
        last = self.messages[-1] if self.messages else None
        if last is not None and last["role"] == "user":
            last["parts"] += parts
        else:
            self.messages.append({"role": "user", "parts": list(parts)})

    def system_entry(self, text):
        # Templates generally reject system messages after the first turn, so a
        # mid-conversation system entry is delivered as text in the user turn.
        if not self.messages:
            self.system.append(text)
        else:
            self.user([{"type": "text", "text": text}])

    def assistant(self, text, reasoning, calls):
        last = self.messages[-1] if self.messages else None
        if last is not None and last["role"] == "assistant" and not last["tool_calls"]:
            last["parts"] += [{"type": "text", "text": text}] if text else []
            last["reasoning"] += reasoning
            last["tool_calls"] += calls
        else:
            self.messages.append(
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "text": text}] if text else [],
                    "reasoning": reasoning,
                    "tool_calls": calls,
                }
            )

    def tool(self, call_id, content):
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

    def render(self):
        rendered = []
        if self.system:
            rendered.append({"role": "system", "content": BLOCK_SEPARATOR.join(self.system)})
        for message in self.messages:
            if message["role"] == "tool":
                rendered.append(message)
            elif message["role"] == "user":
                rendered.append({"role": "user", "content": join_parts(message["parts"])})
            else:
                entry = {"role": "assistant", "content": join_parts(message["parts"]) or None}
                if message["reasoning"]:
                    entry["reasoning_content"] = message["reasoning"]
                if message["tool_calls"]:
                    entry["tool_calls"] = message["tool_calls"]
                rendered.append(entry)
        return rendered


def adapt_messages(data, vision):
    """Returns chat messages and whether the final assistant turn is a prefill."""

    conversation = Conversation(system_text(data.system))
    pending = []
    for index, message in enumerate(data.messages):
        where = f"messages.{index}"
        blocks = (
            [TextBlock(text=message.content)]
            if isinstance(message.content, str)
            else message.content
        )
        if message.role == "system":
            text = [b.text for b in blocks if isinstance(b, TextBlock)]
            if len(text) != len(blocks):
                raise MessagesRequestError(f"{where}: System messages may only contain text")
            conversation.system_entry(BLOCK_SEPARATOR.join(text))
            continue

        if message.role == "assistant":
            if pending:
                raise MessagesRequestError(
                    f"{where}: `tool_use` ids were found without `tool_result` blocks "
                    f"immediately after: {', '.join(pending)}. Each `tool_use` block must "
                    "have a corresponding `tool_result` block in the next message."
                )
            texts, reasoning, calls = [], "", []
            for block in blocks:
                if isinstance(block, TextBlock):
                    texts.append(block.text)
                elif isinstance(block, ThinkingBlock):
                    reasoning += block.thinking
                elif isinstance(block, ToolUseBlock):
                    pending.append(block.id)
                    calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {"name": block.name, "arguments": json.dumps(block.input)},
                        }
                    )
                elif isinstance(block, RedactedThinkingBlock) or _droppable(block):
                    continue
                else:
                    raise MessagesRequestError(
                        f"{where}: Unsupported content block type in assistant messages: "
                        f"{block.type}"
                    )
            conversation.assistant(BLOCK_SEPARATOR.join(texts), reasoning, calls)
            continue

        results = [block for block in blocks if isinstance(block, ToolResultBlock)]
        for block in results:
            if block.tool_use_id not in pending:
                raise MessagesRequestError(
                    f"{where}: unexpected `tool_use_id` found in `tool_result` blocks: "
                    f"{block.tool_use_id}. Each `tool_result` block must have a corresponding "
                    "`tool_use` block in the previous message."
                )
            pending.remove(block.tool_use_id)
            content = block.content
            if isinstance(content, str):
                content = [TextBlock(text=content)] if content else []
            parts = content_parts(content, vision, where)
            conversation.tool(block.tool_use_id, join_parts(parts))
        if pending:
            raise MessagesRequestError(
                f"{where}: `tool_use` ids were found without `tool_result` blocks immediately "
                f"after: {', '.join(pending)}. Each `tool_use` block must have a corresponding "
                "`tool_result` block in the next message."
            )
        others = [block for block in blocks if not isinstance(block, ToolResultBlock)]
        conversation.user(content_parts(others, vision, where))

    messages = conversation.render()
    prefill = bool(messages) and messages[-1]["role"] == "assistant"
    if prefill:
        final = messages[-1]
        if pending:
            raise MessagesRequestError(
                "The final assistant message cannot end with `tool_use` blocks without results"
            )
        if not isinstance(final["content"], str):
            # A thinking-only turn has nothing to continue; generate a fresh turn.
            messages.pop()
            prefill = False
        elif final["content"] != final["content"].rstrip():
            raise MessagesRequestError(
                "messages: final assistant content cannot end with trailing whitespace"
            )
    return messages, prefill


def adapt_request(data, vision=False):
    """Build the internal chat request plus the tool policy used to check model output."""

    tools = ToolSet(data)
    messages, prefill = adapt_messages(data, vision)
    if not any(message["role"] != "system" for message in messages):
        raise MessagesRequestError("messages: at least one user or assistant turn is required")

    thinking = data.thinking
    thinking_on = isinstance(thinking, (ThinkingEnabled, ThinkingAdaptive))
    if thinking_on and tools.required:
        raise MessagesRequestError("Thinking may not be enabled when tool_choice forces tool use.")

    schema = None
    config = data.output_config
    if config is not None and config.format is not None:
        schema = config.format.schema_
        _schema_validator(schema)
        if tools.chat_tools:
            raise MessagesRequestError(
                "output_config.format: Structured outputs cannot be combined with tools here"
            )

    params = {
        name: getattr(data, name)
        for name in ("temperature", "top_p", "top_k", "max_tokens")
        if getattr(data, name, None) is not None
    }
    if getattr(data, "stop_sequences", None):
        params["stop"] = data.stop_sequences
    if isinstance(thinking, ThinkingEnabled):
        params["reasoning_budget_tokens"] = thinking.budget_tokens
    if config is not None and config.effort is not None:
        params["reasoning_effort"] = EFFORTS[config.effort]

    if tools.required:
        names = ", ".join(sorted(tools.allowed))
        instruction = f"You must call one of these tools this turn: {names}."
        if messages[0]["role"] == "system":
            messages[0]["content"] += BLOCK_SEPARATOR + instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})

    chat = ChatCompletionRequest(
        messages=messages,
        tools=tools.chat_tools or None,
        n=1,
        tool_choice="auto",
        json_schema=schema,
        enable_thinking=thinking_on,
        add_generation_prompt=not prefill,
        continue_final_message=prefill,
        **params,
    )
    chat._fail_on_grammar_error = schema is not None
    return chat, tools
