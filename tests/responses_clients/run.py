"""Run SDK protocol smoke tests; --codex also edits a disposable file using Codex.

The backend is deterministic. Passing these tests does not qualify model quality.
"""
import argparse
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}/v1"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(root), str(root / "tests"))),
        "TABBY_TEST_PORT": str(port),
        "TABBY_TEST_URL": base_url,
    }
    with tempfile.TemporaryDirectory(prefix="tabby-responses-") as directory:
        work = Path(directory)
        with (work / "server.log").open("w+") as log:
            server = subprocess.Popen(
                [sys.executable, "tests/responses_mock_server.py"], cwd=root,
                env=env, stdout=log, stderr=log,
            )
            try:
                for _ in range(100):
                    if server.poll() is not None:
                        raise RuntimeError("Fixture server exited during startup")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                            break
                    except OSError:
                        time.sleep(0.1)
                else:
                    raise TimeoutError("Fixture server did not start")
                subprocess.run(
                    [shutil.which("npm") or "npm", "test", "--prefix", "tests/responses_clients"],
                    cwd=root, env=env, check=True, timeout=30,
                )
                if args.codex:
                    (work / "fixture.txt").write_text("before\n")
                    command = [shutil.which("codex") or "codex", "exec", "--ignore-user-config",
                               "--ephemeral", "--skip-git-repo-check", "-C", str(work),
                               "-s", "workspace-write"]
                    options = [
                        'model_provider="tabby_fixture"', 'model="test-model"',
                        'model_providers.tabby_fixture={name="Tabby fixture",base_url="' + base_url +
                        '",wire_api="responses",requires_openai_auth=false,supports_websockets=false,'
                        'request_max_retries=0,stream_max_retries=0}',
                        'web_search="disabled"', 'model_reasoning_summary="none"',
                        'model_supports_reasoning_summaries=false', 'approval_policy="never"',
                        'features.multi_agent=false', 'features.apps=false',
                        'features.skill_search=false', 'features.tool_suggest=false',
                    ]
                    for option in options:
                        command.extend(("-c", option))
                    command.append("CODEX_PROTOCOL_SMOKE: Read fixture.txt, change before to after, "
                                   "then report completion.")
                    result = subprocess.run(command, stdin=subprocess.DEVNULL, cwd=work,
                                            capture_output=True, text=True, timeout=60)
                    if result.returncode or "CODEX_PROTOCOL_SMOKE_OK" not in result.stdout:
                        raise RuntimeError(result.stdout + result.stderr)
                    if (work / "fixture.txt").read_text() != "after\n":
                        raise AssertionError("Codex did not edit the fixture")
                    print("Codex: read, tool-driven edit, result replay, and final answer passed.")
            except BaseException:
                log.flush()
                log.seek(0)
                print(log.read()[-4000:], file=sys.stderr)
                raise
            finally:
                server.terminate()
                server.wait(timeout=5)


if __name__ == "__main__":
    main()
