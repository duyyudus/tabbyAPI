"""Anthropic Messages API request contract, served statelessly against the loaded model.

Unknown fields are ignored rather than rejected: Claude Code adds request fields and
beta options with each release (cache_control, context_management, defer_loading, ...),
and a gateway is expected to treat them as open lists. Every field that changes the
generated result is declared and validated.
"""

from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    model_validator,
)


class WireModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _tagged(known):
    """Route a block to its model by type, sending anything unrecognized to "other"."""

    def discriminate(value):
        kind = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
        return kind if kind in known else "other"

    return Discriminator(discriminate)


class TextBlock(WireModel):
    type: Literal["text"] = "text"
    text: str


class Base64ImageSource(WireModel):
    type: Literal["base64"]
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str


class URLSource(WireModel):
    type: Literal["url"]
    url: str


class FileSource(WireModel):
    type: Literal["file"]
    file_id: str


class ImageBlock(WireModel):
    type: Literal["image"] = "image"
    source: Annotated[Union[Base64ImageSource, URLSource, FileSource], Field(discriminator="type")]


class Base64PDFSource(WireModel):
    type: Literal["base64"]
    media_type: Literal["application/pdf"]
    data: str


class PlainTextSource(WireModel):
    type: Literal["text"]
    media_type: Literal["text/plain"] = "text/plain"
    data: str


class ContentBlockSource(WireModel):
    type: Literal["content"]
    content: Union[
        str,
        list[
            Annotated[
                Union[Annotated[TextBlock, Tag("text")], Annotated[ImageBlock, Tag("image")]],
                _tagged({"text", "image"}),
            ]
        ],
    ]


class DocumentBlock(WireModel):
    type: Literal["document"] = "document"
    source: Annotated[
        Union[Base64PDFSource, PlainTextSource, ContentBlockSource, URLSource, FileSource],
        Field(discriminator="type"),
    ]
    title: str | None = None
    context: str | None = None


class SearchResultBlock(WireModel):
    type: Literal["search_result"] = "search_result"
    source: str
    title: str
    content: list[TextBlock]


class ThinkingBlock(WireModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    # Signatures are never verified here, so a missing one is not an error.
    signature: str = ""


class RedactedThinkingBlock(WireModel):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


class ToolUseBlock(WireModel):
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(min_length=1, max_length=200)
    input: dict


class OtherBlock(WireModel):
    """Server tool blocks, tool references and future types; the adapter decides."""

    model_config = ConfigDict(extra="allow")

    type: str


ToolResultContent = Annotated[
    Union[
        Annotated[TextBlock, Tag("text")],
        Annotated[ImageBlock, Tag("image")],
        Annotated[DocumentBlock, Tag("document")],
        Annotated[SearchResultBlock, Tag("search_result")],
        Annotated[OtherBlock, Tag("other")],
    ],
    _tagged({"text", "image", "document", "search_result"}),
]


class ToolResultBlock(WireModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    content: Union[str, list[ToolResultContent]] = ""
    is_error: bool | None = None


ContentBlock = Annotated[
    Union[
        Annotated[TextBlock, Tag("text")],
        Annotated[ImageBlock, Tag("image")],
        Annotated[DocumentBlock, Tag("document")],
        Annotated[SearchResultBlock, Tag("search_result")],
        Annotated[ThinkingBlock, Tag("thinking")],
        Annotated[RedactedThinkingBlock, Tag("redacted_thinking")],
        Annotated[ToolUseBlock, Tag("tool_use")],
        Annotated[ToolResultBlock, Tag("tool_result")],
        Annotated[OtherBlock, Tag("other")],
    ],
    _tagged(
        {
            "text",
            "image",
            "document",
            "search_result",
            "thinking",
            "redacted_thinking",
            "tool_use",
            "tool_result",
        }
    ),
]


class Message(WireModel):
    # "system" entries may be appended mid-conversation by Claude Code.
    role: Literal["user", "assistant", "system"]
    content: Union[str, list[ContentBlock]]


class CustomTool(WireModel):
    type: Literal["custom"] | None = None
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,128}$")
    description: str | None = None
    input_schema: dict
    strict: bool | None = None


class ServerTool(WireModel):
    """Anthropic-defined tools (web_search_*, bash_*, ...) cannot run locally."""

    model_config = ConfigDict(extra="allow")

    type: str
    name: str | None = None


Tool = Annotated[
    Union[Annotated[CustomTool, Tag("custom")], Annotated[ServerTool, Tag("server")]],
    Discriminator(
        lambda value: (
            "custom"
            if (value.get("type") if isinstance(value, dict) else value.type) in (None, "custom")
            else "server"
        )
    ),
]


class ToolChoiceAuto(WireModel):
    type: Literal["auto"]
    disable_parallel_tool_use: bool = False


class ToolChoiceAny(WireModel):
    type: Literal["any"]
    disable_parallel_tool_use: bool = False


class ToolChoiceTool(WireModel):
    type: Literal["tool"]
    name: str
    disable_parallel_tool_use: bool = False


class ToolChoiceNone(WireModel):
    type: Literal["none"]
    disable_parallel_tool_use: bool = False


ToolChoice = Annotated[
    Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool, ToolChoiceNone],
    Field(discriminator="type"),
]


class ThinkingEnabled(WireModel):
    type: Literal["enabled"]
    budget_tokens: int = Field(ge=1024)
    display: Literal["summarized", "omitted"] | None = None


class ThinkingAdaptive(WireModel):
    type: Literal["adaptive"]
    display: Literal["summarized", "omitted"] | None = None


class ThinkingDisabled(WireModel):
    type: Literal["disabled"]


Thinking = Annotated[
    Union[ThinkingEnabled, ThinkingAdaptive, ThinkingDisabled], Field(discriminator="type")
]


class JsonOutputFormat(WireModel):
    type: Literal["json_schema"]
    schema_: dict = Field(alias="schema")


class OutputConfig(WireModel):
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    format: JsonOutputFormat | None = None


class CountTokensRequest(WireModel):
    """Fields shared by /v1/messages and /v1/messages/count_tokens."""

    model: str
    messages: list[Message] = Field(min_length=1)
    system: Union[str, list[TextBlock], None] = None
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    thinking: Thinking | None = None
    output_config: OutputConfig | None = None


class MessagesRequest(CountTokensRequest):
    max_tokens: int = Field(ge=1)
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=1)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def check_thinking_budget(self):
        if isinstance(self.thinking, ThinkingEnabled) and (
            self.thinking.budget_tokens >= self.max_tokens
        ):
            raise ValueError("`max_tokens` must be greater than `thinking.budget_tokens`")
        return self
