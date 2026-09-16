"""Stateless Anthropic Messages endpoints with endpoint-local Anthropic error handling."""

import re

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from sse_starlette import EventSourceResponse, ServerSentEvent

from common import model
from common.auth import check_api_key
from common.errors import ContextLengthHTTPException
from common.networking import DisconnectHandler, get_sse_ping_interval
from common.tabby_config import config
from endpoints.Anthropic.types.messages import CountTokensRequest, MessagesRequest
from endpoints.Anthropic.utils.messages import (
    MessageGenerationError,
    collect_message,
    generate_message_events,
    stream_message,
)
from endpoints.Anthropic.utils.messages_input import MessagesRequestError, adapt_request
from endpoints.OAI.responses_router import describe_validation_error
from endpoints.OAI.utils.chat_completion import apply_chat_template
from endpoints.OAI.utils.common_ import load_inline_model
from endpoints.OAI.utils.tools import is_supported_format

ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    529: "overloaded_error",
}


def error_response(message, status=400):
    kind = ERROR_TYPES.get(status, "api_error" if status >= 500 else "invalid_request_error")
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": kind, "message": message}},
    )


class MessagesRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def wrapped(request):
            try:
                return await handler(request)
            except RequestValidationError as exc:
                message, param = describe_validation_error(exc)
                # Anthropic addresses fields as messages.0.content, not messages[0].content.
                param = re.sub(r"\[(\d+)\]", r".\1", param or "").lstrip(".")
                return error_response(f"{param}: {message}" if param else message)
            except MessagesRequestError as exc:
                return error_response(str(exc))
            except ContextLengthHTTPException as exc:
                # Claude Code recognizes this wording and offers to compact the conversation.
                return error_response(f"prompt is too long: {exc.detail}")
            except MessageGenerationError as exc:
                return error_response(str(exc), status=500)
            except HTTPException as exc:
                return error_response(str(exc.detail), status=exc.status_code)

        return wrapped


router = APIRouter(route_class=MessagesRoute)


def count_prompt_tokens(prompt, embeddings):
    return len(model.container.encode_tokens(prompt, embeddings=embeddings))


def anthropic_ping():
    return ServerSentEvent(event="ping", data='{"type": "ping"}')


@router.post("/v1/messages", dependencies=[Depends(check_api_key)])
async def messages_request(request: Request, data: MessagesRequest):
    # Share the model-load serialization used by the existing OAI endpoints.
    from endpoints.OAI.router import load_lock

    if data.stream and config.developer.disable_request_streaming:
        raise MessagesRequestError("Streaming is disabled on this server")
    async with load_lock:
        await load_inline_model(data.model, request)
        await model.check_model_container()
        container = model.container
        model_name = container.model_dir.name
    if container.prompt_template is None:
        raise MessagesRequestError("The loaded model requires a chat template")
    params, tools = adapt_request(data, vision=container.use_vision)
    if tools.chat_tools and not (
        container.harmony or container.muse_glimmer or is_supported_format(container.tool_format)
    ):
        raise MessagesRequestError("tools: Configure a supported tool_format for this model")
    prompt, embeddings = await apply_chat_template(params)
    model.check_context_length(prompt, params, embeddings)
    input_tokens = count_prompt_tokens(prompt, embeddings)
    disconnect = DisconnectHandler(request, "messages")
    try:
        await disconnect.poll()
        events = generate_message_events(
            data,
            params,
            tools,
            prompt,
            embeddings,
            input_tokens,
            request.state.id,
            model_name,
            disconnect,
        )
        if data.stream:
            return EventSourceResponse(
                stream_message(events),
                ping=get_sse_ping_interval(),
                ping_message_factory=anthropic_ping,
            )
        return await collect_message(events)
    except BaseException:
        await disconnect.cleanup()
        raise


@router.post("/v1/messages/count_tokens", dependencies=[Depends(check_api_key)])
async def count_tokens_request(data: CountTokensRequest):
    # Counting never loads a model: it measures the prompt for the one being served.
    await model.check_model_container()
    container = model.container
    if container.prompt_template is None:
        raise MessagesRequestError("The loaded model requires a chat template")
    params, _ = adapt_request(data, vision=container.use_vision)
    prompt, embeddings = await apply_chat_template(params)
    return {"input_tokens": count_prompt_tokens(prompt, embeddings)}
