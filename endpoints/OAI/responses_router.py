"""Stateless Responses endpoint with endpoint-local OpenAI error handling."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from sse_starlette import EventSourceResponse

from common import model
from common.auth import check_api_key
from common.errors import ContextLengthHTTPException
from common.networking import DisconnectHandler, get_sse_ping_interval
from common.tabby_config import config
from endpoints.OAI.types.responses import ResponsesRequest, ResponseObject
from endpoints.OAI.utils.chat_completion import apply_chat_template
from endpoints.OAI.utils.common_ import load_inline_model
from endpoints.OAI.utils.responses import (
    collect_response,
    generate_response_events,
    stream_response,
)
from endpoints.OAI.utils.responses_input import ResponseRequestError, adapt_request
from endpoints.OAI.utils.tools import is_supported_format


def param_path(location, body):
    """Rebuild a validation path in OpenAI's param notation, e.g. input[0].content[0].detail.

    Pydantic reports union branches and discriminator tags as synthetic segments the
    client never sent ("str", "list[tagged-union[...]]", "message", "input_image").
    Walking the submitted body drops exactly those: a real field addresses into the
    body, a synthetic one does not.
    """
    parts = []
    node = body
    segments = [segment for segment in location if segment != "body"]
    for index, segment in enumerate(segments):
        if isinstance(segment, int):
            if isinstance(node, list) and -len(node) <= segment < len(node):
                node = node[segment]
                parts.append(f"[{segment}]")
        elif isinstance(node, dict) and segment in node:
            node = node[segment]
            parts.append(segment)
        elif (
            index == len(segments) - 1
            and isinstance(node, dict)
            and "[" not in segment
        ):
            # A required field is absent from the body, yet still names the fault.
            parts.append(segment)
    return parts


def error_message(error):
    """Pydantic prefixes a raised ValueError; the reason alone is the message."""
    message = error["msg"]
    prefix = "Value error, "
    return message[len(prefix):] if message.startswith(prefix) else message


def describe_validation_error(exc):
    """Pick the most specific failure and render its param.

    `input` is `str | list[InputItem]`, so any fault inside an item also fails the
    string branch; that shallow error sorts first and would otherwise be reported
    as `input.str`. The deepest path names the field the client actually got wrong.
    """
    errors = exc.errors()
    body = getattr(exc, "body", None)
    if not isinstance(body, (dict, list)):
        # An unparsable body has no field to blame; its location is an offset.
        error = errors[0]
        named = [p for p in error["loc"] if isinstance(p, str) and p != "body"]
        return error_message(error), ".".join(named) or None
    ranked = [(param_path(error["loc"], body), error) for error in errors]
    parts, error = max(ranked, key=lambda pair: len(pair[0]))
    param = ""
    for part in parts:
        param += part if part.startswith("[") else (f".{part}" if param else part)
    return error_message(error), param or None


def error_response(message, status=400, param=None, code="invalid_value"):
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "server_error",
                "param": param,
                "code": code,
            }
        },
    )


class ResponsesRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def wrapped(request):
            try:
                return await handler(request)
            except RequestValidationError as exc:
                message, param = describe_validation_error(exc)
                return error_response(message, param=param)
            except ResponseRequestError as exc:
                return error_response(str(exc), param=exc.param, code=exc.code)
            except ContextLengthHTTPException as exc:
                return error_response(str(exc.detail), code="context_length_exceeded")
            except HTTPException as exc:
                return error_response(str(exc.detail), status=exc.status_code)

        return wrapped


router = APIRouter(route_class=ResponsesRoute)


@router.post("/v1/responses", dependencies=[Depends(check_api_key)], response_model=ResponseObject)
async def responses_request(request: Request, data: ResponsesRequest):
    # Share the model-load serialization used by the existing OAI endpoints.
    from endpoints.OAI.router import load_lock

    if data.stream and config.developer.disable_request_streaming:
        raise ResponseRequestError("Streaming is disabled on this server", "stream")
    async with load_lock:
        if data.model:
            await load_inline_model(data.model, request)
        await model.check_model_container()
        container = model.container
        model_name = container.model_dir.name
    if container.prompt_template is None:
        raise ResponseRequestError("The loaded model requires a chat template")
    params, tools = adapt_request(data, vision=container.use_vision)
    if tools.allowed and not (
        container.harmony or container.muse_glimmer or is_supported_format(container.tool_format)
    ):
        raise ResponseRequestError("Configure a supported tool_format for this model", "tools")
    prompt, embeddings = await apply_chat_template(params)
    model.check_context_length(prompt, params, embeddings)
    disconnect = DisconnectHandler(request, "responses")
    try:
        await disconnect.poll()
        events = generate_response_events(
            data,
            params,
            tools,
            prompt,
            embeddings,
            request.state.id,
            model_name,
            disconnect,
        )
        if data.stream:
            return EventSourceResponse(stream_response(events), ping=get_sse_ping_interval())
        return await collect_response(events)
    except BaseException:
        await disconnect.cleanup()
        raise
