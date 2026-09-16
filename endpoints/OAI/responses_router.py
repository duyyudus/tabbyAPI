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
                error = exc.errors()[0]
                param = ".".join(str(p) for p in error["loc"] if p != "body")
                return error_response(error["msg"], param=param)
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
