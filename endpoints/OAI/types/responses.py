"""The supported stateless Responses wire contract (independent of chat types)."""

from time import time
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
)


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# A client may spell "unset" as an explicit null; treat it as the field's default
# rather than a rejection, since null carries no instruction to honor.
Description = Annotated[str, BeforeValidator(lambda value: "" if value is None else value)]
Parameters = Annotated[
    dict,
    BeforeValidator(
        lambda value: {"type": "object", "properties": {}} if value is None else value
    ),
]
# Replayed assistant turns can carry annotations and logprobs produced elsewhere.
# They are accepted and dropped: none are generated here, so none are echoed back.
Dropped = Annotated[list, BeforeValidator(lambda value: [])]


class InputText(WireModel):
    type: Literal["input_text"]
    text: str


class OutputText(WireModel):
    type: Literal["output_text"] = "output_text"
    text: str = ""
    annotations: Dropped = Field(default_factory=list)
    logprobs: Dropped = Field(default_factory=list)


class InputImage(WireModel):
    type: Literal["input_image"]
    image_url: str
    # Resolution control is not implemented; every hint is accepted and ignored.
    detail: Literal["auto", "low", "high"] | None = "auto"


Content = Annotated[InputText | OutputText | InputImage, Field(discriminator="type")]


class InputMessage(WireModel):
    type: Literal["message"] = "message"
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[Content]
    id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    phase: Literal["commentary", "final_answer"] | None = None


class FunctionCall(WireModel):
    type: Literal["function_call"] = "function_call"
    id: str | None = None
    call_id: str
    name: str
    arguments: str
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class CustomCall(WireModel):
    type: Literal["custom_tool_call"] = "custom_tool_call"
    id: str | None = None
    call_id: str
    name: str
    input: str
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class ToolResult(WireModel):
    type: Literal["function_call_output", "custom_tool_call_output"]
    call_id: str
    output: str | list[Annotated[InputText | InputImage, Field(discriminator="type")]]
    id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class ReasoningItem(WireModel):
    """A reasoning item replayed by a client: accepted for wire compatibility, never used.

    No reasoning items are produced, so nothing here can round-trip meaningfully. Field
    drift in the upstream item is tolerated because the whole item is dropped on input.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["reasoning"] = "reasoning"
    id: str | None = None
    summary: list = Field(default_factory=list)
    content: list = Field(default_factory=list)
    encrypted_content: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None


InputItem = Annotated[
    InputMessage | FunctionCall | CustomCall | ToolResult | ReasoningItem,
    Field(discriminator="type"),
]


class CustomFormat(WireModel):
    type: Literal["text", "grammar"] = "text"
    syntax: Literal["lark"] | None = None
    definition: str | None = None


class FunctionTool(WireModel):
    type: Literal["function"]
    name: str = Field(min_length=1)
    description: Description = ""
    parameters: Parameters = Field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool | None = None


class CustomTool(WireModel):
    type: Literal["custom"]
    name: str = Field(min_length=1)
    description: Description = ""
    format: CustomFormat = Field(default_factory=CustomFormat)


NestedToolDefinition = Annotated[FunctionTool | CustomTool, Field(discriminator="type")]


class NamespaceTool(WireModel):
    type: Literal["namespace"]
    name: str = Field(min_length=1)
    description: Description = ""
    tools: list[NestedToolDefinition] = Field(default_factory=list)


class UnsupportedTool(WireModel):
    """A hosted or unknown tool type: accepted for wire compatibility, never executed."""

    type: Literal["__unsupported__"] = "__unsupported__"
    definition: dict

    @model_serializer(mode="plain")
    def as_wire_definition(self):
        return self.definition


# Tool kinds the server can present to the model; anything else is held as an
# UnsupportedTool and ignored.
SUPPORTED_TOOL_TYPES = frozenset({"function", "custom", "namespace"})


def _tag_unsupported_tools(value):
    if isinstance(value, list):
        return [
            {"type": "__unsupported__", "definition": item}
            if isinstance(item, dict)
            and isinstance(item.get("type"), str)
            and item["type"] not in SUPPORTED_TOOL_TYPES
            else item
            for item in value
        ]
    return value


ToolDefinition = Annotated[
    FunctionTool | CustomTool | NamespaceTool | UnsupportedTool, Field(discriminator="type")
]

ToolList = Annotated[list[ToolDefinition], BeforeValidator(_tag_unsupported_tools)]


class NamedToolChoice(WireModel):
    type: Literal["function", "custom"]
    name: str


class AllowedTools(WireModel):
    type: Literal["allowed_tools"]
    mode: Literal["auto", "required"]
    tools: list[NamedToolChoice] = Field(min_length=1)


class TextFormat(WireModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    name: str | None = None
    schema_: dict | None = Field(default=None, alias="schema")
    strict: bool | None = None
    description: str | None = None


class TextOptions(WireModel):
    format: TextFormat = Field(default_factory=TextFormat)
    verbosity: Literal["low", "medium", "high"] | None = None


class ReasoningOptions(WireModel):
    effort: str | None = None
    # A summary request is a request for available data, not a demand to invent one:
    # every mode is accepted and none is honored, because reasoning stays internal.
    summary: Literal["auto", "concise", "detailed", "none"] | None = None


UNSUPPORTED = {
    "store": "this server is stateless and never persists a response, so store must be false",
    "background": "there is no job queue here, so background must be false",
    "previous_response_id": "no response history is kept; replay the earlier items in input",
    "conversation": "Conversations are not implemented; replay the earlier items in input",
    "prompt": "server-side prompt templates are not implemented; send instructions instead",
    "max_tool_calls": "a tool-call budget is not enforced here; omit it rather than rely on it",
    "context_management": "automatic context compaction is not implemented",
    "moderation": "no moderation model runs here, and accepting this would imply "
                  "input and output filtering that never happens",
    "truncation": "an oversized context fails instead of being truncated, "
                  "so truncation must be disabled",
}


class ResponsesRequest(WireModel):
    model: str | None = None
    input: str | list[InputItem] = Field(default_factory=list)
    instructions: str | None = None
    stream: bool = False
    store: bool = False
    background: bool = False
    previous_response_id: str | None = None
    conversation: str | dict | None = None
    prompt: dict | None = None
    max_tool_calls: int | None = None
    context_management: list | dict | None = None
    moderation: dict | None = None
    max_output_tokens: int | None = Field(default=None, ge=16)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    tools: ToolList = Field(default_factory=list)
    tool_choice: Literal["auto", "none", "required"] | NamedToolChoice | AllowedTools = "auto"
    parallel_tool_calls: bool = True
    text: TextOptions = Field(default_factory=TextOptions)
    reasoning: ReasoningOptions | None = None
    metadata: dict[str, str] = Field(default_factory=dict, max_length=16)
    # Advisory routing/telemetry fields; local KV caching remains backend-managed.
    # Each is recorded and ignored: none can change what this server generates, so
    # rejecting them would only break clients that send them by default.
    prompt_cache_key: str | None = None
    client_metadata: dict = Field(default_factory=dict)
    prompt_cache_retention: str | None = None
    prompt_cache_options: dict | None = None
    service_tier: str | None = None
    user: str | None = None
    safety_identifier: str | None = None
    # Obfuscation padding is a mitigation for OpenAI's network, not this one.
    stream_options: dict | None = None
    truncation: str = "disabled"
    # An include is a request for available data, not a demand to invent it: any
    # include is accepted, and one whose data this server never produces yields
    # nothing. No reasoning items are returned, so no encrypted content exists.
    include: list[str] = Field(default_factory=list)
    # Likewise a request for token logprobs, which this server does not produce.
    top_logprobs: int | None = Field(default=None, ge=0, le=20)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value):
        for key, item in value.items():
            if len(key) > 64:
                raise ValueError("Metadata keys must be at most 64 characters")
            if len(item) > 512:
                raise ValueError("Metadata values must be at most 512 characters")
        return value

    @field_validator(
        "store", "background", "previous_response_id", "conversation", "prompt",
        "max_tool_calls", "context_management", "moderation", "truncation",
        mode="after",
    )
    @classmethod
    def reject_unsupported(cls, value, info):
        """Refuse options whose effect cannot be reproduced, naming the reason.

        Silently ignoring any of these would change the answer, or what the caller
        believes happened to it, without a word. Unlike an advisory hint, that is
        worth a failed request.
        """
        if value != cls.model_fields[info.field_name].get_default():
            raise ValueError(UNSUPPORTED[info.field_name])
        return value

    @field_validator("input", mode="before")
    @classmethod
    def shorthand_messages(cls, value):
        if isinstance(value, list):
            return [
                {"type": "message", **item} if isinstance(item, dict) and "role" in item else item
                for item in value
            ]
        return value


class OutputMessage(WireModel):
    type: Literal["message"] = "message"
    id: str = Field(default_factory=lambda: "msg_" + uuid4().hex)
    role: Literal["assistant"] = "assistant"
    status: Literal["in_progress", "completed", "incomplete"] = "in_progress"
    content: list[OutputText] = Field(default_factory=list)


class InputTokenDetails(WireModel):
    cached_tokens: int = 0


class OutputTokenDetails(WireModel):
    # No exact reasoning attribution is available from the backend.
    reasoning_tokens: int | None = None


class ResponseUsage(WireModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: InputTokenDetails = Field(default_factory=InputTokenDetails)
    output_tokens_details: OutputTokenDetails = Field(default_factory=OutputTokenDetails)

    @model_serializer(mode="wrap")
    def serialize_usage(self, handler):
        result = handler(self)
        if self.output_tokens_details.reasoning_tokens is None:
            result.pop("output_tokens_details", None)
        return result


class ResponseObject(WireModel):
    id: str = Field(default_factory=lambda: "resp_" + uuid4().hex)
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time()))
    completed_at: int | None = None
    status: Literal["in_progress", "completed", "incomplete", "failed"] = "in_progress"
    error: dict | None = None
    incomplete_details: dict | None = None
    output: list[OutputMessage | FunctionCall | CustomCall] = Field(default_factory=list)
    model: str
    usage: ResponseUsage | None = None
    store: bool = False
    background: bool = False
    previous_response_id: None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    parallel_tool_calls: bool = True
    tool_choice: str | NamedToolChoice | AllowedTools = "auto"
    tools: ToolList = Field(default_factory=list)
    text: TextOptions = Field(default_factory=TextOptions)
    reasoning: ReasoningOptions | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    truncation: Literal["disabled"] = "disabled"


class ResponseEvent(BaseModel):
    """Common envelope; individual event payloads vary by type."""

    model_config = ConfigDict(extra="allow")
    type: str
    sequence_number: int
