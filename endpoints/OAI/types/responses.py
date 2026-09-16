"""The supported stateless Responses wire contract (independent of chat types)."""

from time import time
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputText(WireModel):
    type: Literal["input_text"]
    text: str


class OutputText(WireModel):
    type: Literal["output_text"] = "output_text"
    text: str = ""
    annotations: list = Field(default_factory=list, max_length=0)
    logprobs: list = Field(default_factory=list, max_length=0)


class InputImage(WireModel):
    type: Literal["input_image"]
    image_url: str
    detail: Literal["auto"] = "auto"


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


InputItem = Annotated[
    InputMessage | FunctionCall | CustomCall | ToolResult, Field(discriminator="type")
]


class CustomFormat(WireModel):
    type: Literal["text", "grammar"] = "text"
    syntax: Literal["lark"] | None = None
    definition: str | None = None


class FunctionTool(WireModel):
    type: Literal["function"]
    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict = Field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool | None = None


class CustomTool(WireModel):
    type: Literal["custom"]
    name: str = Field(min_length=1)
    description: str = ""
    format: CustomFormat = Field(default_factory=CustomFormat)


ToolDefinition = Annotated[FunctionTool | CustomTool, Field(discriminator="type")]


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
    summary: Literal["none"] | None = None


class ResponsesRequest(WireModel):
    model: str | None = None
    input: str | list[InputItem] = Field(default_factory=list)
    instructions: str | None = None
    stream: bool = False
    store: Literal[False] = False
    background: Literal[False] = False
    previous_response_id: None = None
    conversation: None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    tools: list[ToolDefinition] = Field(default_factory=list)
    tool_choice: Literal["auto", "none", "required"] | NamedToolChoice | AllowedTools = "auto"
    parallel_tool_calls: bool = True
    text: TextOptions = Field(default_factory=TextOptions)
    reasoning: ReasoningOptions | None = None
    metadata: dict[str, str] = Field(default_factory=dict, max_length=16)
    # Advisory routing/telemetry fields; local KV caching remains backend-managed.
    prompt_cache_key: str | None = None
    client_metadata: dict = Field(default_factory=dict)
    truncation: Literal["disabled"] = "disabled"
    # An include is a request for available data, not a demand to invent it.
    include: list[Literal["reasoning.encrypted_content"]] = Field(default_factory=list)

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
    tools: list[ToolDefinition] = Field(default_factory=list)
    text: TextOptions = Field(default_factory=TextOptions)
    reasoning: ReasoningOptions | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    truncation: Literal["disabled"] = "disabled"


class ResponseEvent(BaseModel):
    """Common envelope; individual event payloads vary by type."""

    model_config = ConfigDict(extra="allow")
    type: str
    sequence_number: int
