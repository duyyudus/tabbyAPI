"""Deterministic protocol fixture, never an inference-quality test.

Run: PYTHONPATH=.:tests python tests/responses_mock_server.py
"""
import json
import os
import uvicorn

from common import model
from test_responses import FakeContainer, make_app, finish, tool_text


class ProtocolFixture(FakeContainer):
    async def stream_generate(self, request_id, prompt, params, *args, **kwargs):
        text = " ".join(m.content for m in params.messages if isinstance(m.content, str))
        results = [m for m in params.messages if m.role == "tool"]
        definitions = {t.function.name: t.function for t in params.tools or []}
        if "CODEX_PROTOCOL_SMOKE" in text:
            if not results:
                name = next((n for n in definitions if n in {"exec_command", "shell_command", "shell"}), None)
                if name is None:
                    raise RuntimeError(f"No supported fixture shell tool; received {list(definitions)}")
                key = "cmd" if name == "exec_command" else "command"
                value = ["bash", "-lc", "cat fixture.txt"] if name == "shell" else "cat fixture.txt"
                output = tool_text(name, {key: value})
            elif len(results) == 1:
                if "apply_patch" in definitions and "input" in definitions["apply_patch"].parameters.get("properties", {}):
                    output = tool_text("apply_patch", {"input": "*** Begin Patch\n*** Update File: fixture.txt\n@@\n-before\n+after\n*** End Patch\n"})
                else:
                    name = next(n for n in definitions if n in {"exec_command", "shell_command", "shell"})
                    key = "cmd" if name == "exec_command" else "command"
                    cmd = "printf 'after\\n' > fixture.txt"
                    output = tool_text(name, {key: ["bash", "-lc", cmd] if name == "shell" else cmd})
            else:
                output = "CODEX_PROTOCOL_SMOKE_OK"
        elif results:
            output = "tool result received"
        elif definitions and "use tool" in text:
            name = next(iter(definitions))
            args = {"input": "*** Begin Patch\n*** End Patch\n"} if name == "patch" else {"path": "x"}
            output = tool_text(name, args)
        else:
            output = "hello"
        yield {"text": output}
        yield finish()


model.container = ProtocolFixture()
model.check_context_length = lambda *args, **kwargs: None
app = make_app()


@app.middleware("http")
async def fixture_request_diagnostics(request, call_next):
    if request.method == "POST":
        data = await request.json()
        print("FIXTURE REQUEST", json.dumps({
            "keys": list(data),
            "tools": [{k: t[k] for k in ("type", "name") if k in t}
                      for t in data.get("tools", []) if t.get("type") not in ("function", "custom")],
        }), flush=True)
    return await call_next(request)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("TABBY_TEST_PORT", "18081")))
