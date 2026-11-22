#!/usr/bin/env python3
"""
Find gpu_split and cpu_moe_split_experts for a CPU-offloaded MoE model on this machine's GPUs.

Run it with tabbyAPI's interpreter, from the repository root, with the same environment variables
you serve with (pipeline depth, copy-ahead buffer and attention options change memory use), while
nothing else uses the GPUs:

    CUDA_DEVICE_ORDER=PCI_BUS_ID EXL3_MOE_HOST_CORES=2 \\
        venv/bin/python tools/tune_gpu_split.py --config config_ds.yml

Every trial starts the server with a copy of the config that only changes gpu_split,
cpu_moe_split_experts and the network settings, so the draft model, cache size, chunk size,
batch size and attention options are the ones you will serve with.

1. For the current GPU budgets, it searches for the fewest experts per layer on the CPU that still
   load (fewer is faster for prefill and decode). Each probe is one model load.
2. It loads that layout and sends long test prompts (by default about 64K tokens and one near
   max_seq_len), recording each GPU's peak memory use with nvidia-smi.
3. If a GPU runs out of memory or ends with less free memory than --margin-mib, it lowers that
   GPU's budget and searches again: layers move to later GPUs, or more experts go to the CPU if
   they have no room.

It prints the layout that passed, with each GPU's peak use and the test prompts' speeds, and writes
a JSON record of every trial. Budgets are GiB per GPU in CUDA_DEVICE_ORDER=PCI_BUS_ID order, as in
config.yml; a budget at or above a card's size means all of it, and 0 leaves a card out.
"""

import argparse
import copy
import json
import math
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

from ruamel.yaml import YAML

REPO = pathlib.Path(__file__).resolve().parent.parent


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- GPUs ---------------------------------------------------------------------------------------


def query_gpus(fields: str) -> list[list[str]]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines()]


class MemPoller:
    """Streams nvidia-smi memory readings and keeps each GPU's peak since the last reset."""

    def __init__(self, interval_ms: int = 200):
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits",
             f"--loop-ms={interval_ms}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self.current: dict[int, int] = {}
        self.peak: dict[int, int] = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                i, used = (int(x) for x in line.split(","))
            except ValueError:
                continue
            with self.lock:
                self.current[i] = used
                self.peak[i] = max(self.peak.get(i, 0), used)

    def reset(self):
        with self.lock:
            self.peak = dict(self.current)

    def snapshot(self) -> tuple[dict[int, int], dict[int, int]]:
        with self.lock:
            return dict(self.current), dict(self.peak)

    def close(self):
        self.proc.terminate()


# --- Server -------------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(port: int, path: str, body: dict | None = None, timeout: float = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def flat_log(path: pathlib.Path) -> str:
    """The server log with loguru's line wrapping undone, for pattern matching."""
    return re.sub(r"\s+", " ", path.read_text(errors="replace"))


class Server:
    def __init__(self, base_cfg: dict, workdir: pathlib.Path, name: str):
        self.base_cfg, self.workdir, self.name = base_cfg, workdir, name
        self.proc = None
        self.port = None
        self.log_path = workdir / f"{name}.log"

    def start(self, split: list[int], experts: int):
        cfg = copy.deepcopy(self.base_cfg)
        self.port = free_port()
        net = cfg.setdefault("network", {})
        net.update(host="127.0.0.1", port=self.port, disable_auth=True)
        mdl = cfg.setdefault("model", {})
        mdl.update(gpu_split=list(split), gpu_split_auto=False, cpu_moe_split_experts=experts)
        mdl.pop("cpu_moe_offload_layers", None)
        cfg.setdefault("logging", {})["log_live_status"] = False
        cfg_path = self.workdir / f"{self.name}.yml"
        with open(cfg_path, "w") as f:
            YAML(typ="safe").dump(cfg, f)
        self.proc = subprocess.Popen(
            [sys.executable, "main.py", "--config", str(cfg_path)],
            cwd=REPO, env=os.environ.copy(), start_new_session=True,
            stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT,
        )

    def wait_loaded(self, timeout: float) -> bool:
        """True once the model answers, False if the server exited (a failed load exits)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                http(self.port, "/v1/model", timeout=5)
                return True
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                time.sleep(3)
        raise TimeoutError(f"model did not load within {timeout:.0f} s; see {self.log_path}")

    def stop(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
            except ProcessLookupError:
                pass
        # The expert worker and other children share the process group
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc = None


def wait_gpus_released(baseline: dict[int, int], timeout: float = 180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        used = {int(i): int(u) for i, u in query_gpus("index,memory.used")}
        if all(used[i] <= baseline.get(i, 0) + 512 for i in used):
            return
        time.sleep(2)
    log("warning: GPU memory not back to its starting level after the server exited")


# --- Model facts --------------------------------------------------------------------------------


def find_expert_count(model_dir: pathlib.Path) -> int | None:
    """Routed experts per MoE layer from config.json (ignores drafter-specific keys)."""
    try:
        cfg = json.loads((model_dir / "config.json").read_text())
    except (OSError, ValueError):
        return None
    keys = ("n_routed_experts", "num_experts", "num_local_experts")

    def walk(d):
        if isinstance(d, dict):
            for k in keys:
                if isinstance(d.get(k), int) and d[k] > 1:
                    return d[k]
            for k, v in d.items():
                if "dspark" in k or "mtp" in k or "draft" in k:
                    continue
                r = walk(v)
                if r:
                    return r
        return None

    return walk(cfg)


# --- Test prompts -------------------------------------------------------------------------------


def corpus_text() -> str:
    files = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.py", "*.md"], capture_output=True, text=True
    ).stdout.split()
    if not files:
        files = [str(p.relative_to(REPO)) for p in REPO.rglob("*.py")]
    parts = []
    for f in sorted(files):
        try:
            parts.append(f"\n\n===== {f} =====\n" + (REPO / f).read_text(errors="replace"))
        except OSError:
            pass
    return "".join(parts)


class PromptMaker:
    """Builds uncached prompts of a given token length from this repository's source files."""

    def __init__(self, port: int):
        self.port = port
        self.base = corpus_text()
        sample = self.base[:200_000]
        n = http(port, "/v1/token/encode", {"text": sample}, timeout=120)["length"]
        self.chars_per_token = len(sample) / max(n, 1)

    def make(self, tokens: int) -> tuple[str, int]:
        tail = "\n\nSummarize the text above in one sentence."
        chars = int(tokens * self.chars_per_token * 0.98)
        for _ in range(4):
            body = (self.base * (chars // len(self.base) + 1))[:chars]
            text = f"[{uuid.uuid4().hex}]\n{body}{tail}"
            n = http(self.port, "/v1/token/encode", {"text": text}, timeout=600)["length"]
            if n <= tokens:
                return text, n
            chars = int(chars * tokens / n * 0.995)
        raise RuntimeError("could not build a prompt of the requested length")


# --- Trials -------------------------------------------------------------------------------------


class Tuner:
    def __init__(self, args, base_cfg: dict, gpus: list[dict], workdir: pathlib.Path):
        self.args, self.base_cfg, self.gpus, self.workdir = args, base_cfg, gpus, workdir
        self.baseline = {g["index"]: g["used"] for g in gpus}
        self.trials: list[dict] = []
        self.fit_cache: dict[tuple, bool] = {}
        self.n = 0

    def _server(self) -> Server:
        self.n += 1
        return Server(self.base_cfg, self.workdir, f"trial{self.n:02d}")

    def fits(self, split: list[int], experts: int) -> bool:
        key = (tuple(split), experts)
        if key in self.fit_cache:
            return self.fit_cache[key]
        srv = self._server()
        t0 = time.time()
        srv.start(split, experts)
        try:
            ok = srv.wait_loaded(self.args.load_timeout)
            used = {}
            if ok:
                used = {int(i): int(u) for i, u in query_gpus("index,memory.used")}
                # The drafter's layers may have fewer experts than the model's: take the largest
                m = re.findall(r"CPU split experts[^\[]*\[\d+\.\.\d+\) of (\d+)", flat_log(srv.log_path))
                if m and not self.args.max_experts:
                    self.args.max_experts = max(int(x) for x in m)
        finally:
            srv.stop()
            wait_gpus_released(self.baseline)
        if not ok:
            text = flat_log(srv.log_path)
            if not re.search(r"Insufficient VRAM|OutOfMemoryError|CUDA out of memory", text):
                tail = srv.log_path.read_text(errors="replace")[-3000:]
                raise RuntimeError(f"server exited for another reason ({srv.log_path}):\n{tail}")
        self.fit_cache[key] = ok
        self.trials.append(dict(kind="load", split=list(split), experts=experts, loads=ok,
                                seconds=round(time.time() - t0), used_mib=used, log=str(srv.log_path)))
        log(f"load  split {split}  experts {experts:4d}: {'loads' if ok else 'does not fit'}"
            f"  ({time.time() - t0:.0f} s)")
        return ok

    def min_experts(self, split: list[int], start: int, floor: int = -1) -> int:
        """Fewest CPU experts per layer that load with this split (one load per probe). `floor` is a
        count already known not to fit: lowering a budget never lets fewer experts fit."""
        hi_bound = self.args.max_experts or 1024
        start = max(floor + 1, min(start, hi_bound))
        step = self.args.step
        if self.fits(split, start):
            hi, lo = start, floor
            while hi > lo + 1:
                t = max(lo + 1, hi - step)
                if self.fits(split, t):
                    hi, step = t, step * 2
                else:
                    lo = t
                    break
        else:
            lo = start
            while True:
                hi_bound = self.args.max_experts or hi_bound
                if lo >= hi_bound:
                    raise RuntimeError(f"split {split} does not fit even with every expert on the CPU")
                t = min(hi_bound, lo + step)
                if self.fits(split, t):
                    hi = t
                    break
                lo, step = t, step * 2
        while hi - lo > self.args.resolution:
            mid = (lo + hi) // 2
            if self.fits(split, mid):
                hi = mid
            else:
                lo = mid
        return hi

    def runtime_test(self, split: list[int], experts: int, lengths: list[int]) -> dict:
        srv = self._server()
        srv.start(split, experts)
        poller = MemPoller()
        result = dict(kind="test", split=list(split), experts=experts, prompts=[], ok=True,
                      oom_gpu=None, log=str(srv.log_path))
        try:
            if not srv.wait_loaded(self.args.load_timeout):
                raise RuntimeError(f"layout that loaded before failed to load now; see {srv.log_path}")
            maker = PromptMaker(srv.port)
            peak_all: dict[int, int] = {}
            for target in lengths:
                text, n = maker.make(target)
                poller.reset()
                t0 = time.time()
                rec = dict(target=target, prompt_tokens=n)
                try:
                    r = http(srv.port, "/v1/completions",
                             {"prompt": text, "max_tokens": self.args.decode_tokens,
                              "temperature": 0.0, "top_k": 1}, timeout=7200)
                    u = r.get("usage") or {}
                    ok = bool(r.get("choices")) and "error" not in r
                    rec.update(prefill_tps=u.get("prompt_tokens_per_sec"),
                               decode_tps=u.get("completion_tokens_per_sec"), ok=ok)
                    if not ok:
                        rec["error"] = json.dumps(r)[:200]
                except (urllib.error.HTTPError, urllib.error.URLError, ConnectionError) as e:
                    rec.update(ok=False, error=str(e)[:200])
                time.sleep(1)
                _, peak = poller.snapshot()
                rec.update(seconds=round(time.time() - t0, 1), peak_mib=peak)
                for i, p in peak.items():
                    peak_all[i] = max(peak_all.get(i, 0), p)
                text_log = flat_log(srv.log_path)
                oom = re.findall(r"GPU (\d+) has a total capacity", text_log)
                if not rec["ok"] or oom or "completion aborted" in text_log:
                    result.update(ok=False, oom_gpu=int(oom[-1]) if oom else None)
                    rec["ok"] = False
                result["prompts"].append(rec)
                log(f"test  split {split}  experts {experts:4d}: {n:7d}-token prompt "
                    + (f"ok, prefill {rec.get('prefill_tps')} T/s, decode {rec.get('decode_tps')} T/s"
                       if rec["ok"] else f"FAILED (out of memory on GPU {result['oom_gpu']})"))
                if not rec["ok"]:
                    break
            result["peak_mib"] = peak_all
            result["free_mib"] = {g["index"]: g["total"] - peak_all.get(g["index"], 0) for g in self.gpus}
        finally:
            poller.close()
            srv.stop()
            wait_gpus_released(self.baseline)
        self.trials.append(result)
        return result


# --- Main ---------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="tabbyAPI config to tune (a copy is used per trial)")
    ap.add_argument("--split", help="starting budgets in GiB, e.g. 70,24,24,24,24 (default: the config's "
                    "gpu_split if it has one entry per GPU, else each card's size)")
    ap.add_argument("--experts-start", type=int, help="first expert count to try (default: the config's)")
    ap.add_argument("--max-experts", type=int, help="experts per layer (default: from the model)")
    ap.add_argument("--resolution", type=int, default=1,
                    help="stop the expert search within this many experts (default 1; 2 or 4 save loads)")
    ap.add_argument("--step", type=int, default=8, help="first step of the expert search (default 8)")
    ap.add_argument("--test-tokens", default="65536,max",
                    help="test prompt lengths, comma-separated; 'max' is the longest prompt the cache allows "
                         "(default 65536,max)")
    ap.add_argument("--decode-tokens", type=int, default=64, help="tokens generated after each test prompt")
    ap.add_argument("--margin-mib", type=int, default=512,
                    help="free memory each GPU must keep at the test prompts' peak (default 512; peaks vary "
                         "by a few hundred MiB between runs)")
    ap.add_argument("--max-rounds", type=int, default=6, help="budget adjustments before giving up")
    ap.add_argument("--load-timeout", type=float, default=1800, help="seconds to wait for a model load")
    ap.add_argument("--out", help="JSON record of every trial (default: next to the trial logs)")
    args = ap.parse_args()

    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        log("CUDA_DEVICE_ORDER is not PCI_BUS_ID: setting it, so budgets follow nvidia-smi's order")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        sys.exit("Unset CUDA_VISIBLE_DEVICES: leave a card out with a 0 budget instead")

    cfg_path = pathlib.Path(args.config).resolve()
    base_cfg = YAML(typ="safe").load(cfg_path.read_text())
    mdl = base_cfg.get("model") or {}
    if mdl.get("tensor_parallel"):
        sys.exit("This tool tunes layer splits; tensor_parallel is on in the config")
    if not mdl.get("model_name"):
        sys.exit("The config names no model (model.model_name)")
    model_dir = (REPO / mdl.get("model_dir", "models") / mdl["model_name"]).resolve()

    gpus = [dict(index=int(i), name=n, total=int(t), used=int(u))
            for i, n, t, u in query_gpus("index,name,memory.total,memory.used")]
    busy = [g for g in gpus if g["used"] > 1024]
    if busy:
        sys.exit("GPUs in use: " + ", ".join(f"{g['index']} ({g['used']} MiB)" for g in busy)
                 + ". Stop other GPU work first.")
    for g in gpus:
        log(f"GPU {g['index']}: {g['name']}, {g['total']} MiB")

    if args.split:
        split = [int(float(x)) for x in args.split.split(",")]
    elif isinstance(mdl.get("gpu_split"), list) and len(mdl["gpu_split"]) == len(gpus):
        split = [int(x) for x in mdl["gpu_split"]]
    else:
        split = [round(g["total"] / 1024) for g in gpus]
    if len(split) != len(gpus):
        sys.exit(f"--split has {len(split)} entries for {len(gpus)} GPUs")

    args.max_experts = args.max_experts or find_expert_count(model_dir)
    start = args.experts_start if args.experts_start is not None else (
        mdl.get("cpu_moe_split_experts") or (args.max_experts or 256) // 4)

    max_seq = int(mdl.get("cache_size") or mdl.get("max_seq_len") or 0)
    lengths = []
    for x in args.test_tokens.split(","):
        x = x.strip()
        if x == "max":
            if not max_seq:
                sys.exit("'max' test length needs max_seq_len or cache_size in the config")
            lengths.append(max_seq - args.decode_tokens - 1024)
        elif x:
            lengths.append(int(x))
    lengths = sorted(set(n for n in lengths if n > 0))

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="tune_gpu_split_"))
    out_path = pathlib.Path(args.out) if args.out else workdir / "trials.json"
    log(f"model {mdl['model_name']}, {args.max_experts or '?'} experts per layer; start split {split}, "
        f"experts {start}; test prompts {lengths}; logs in {workdir}")

    tuner = Tuner(args, base_cfg, gpus, workdir)
    used_gpus = [i for i, b in enumerate(split) if b > 0]
    final = None
    floor = -1
    try:
        for rnd in range(args.max_rounds):
            experts = tuner.min_experts(split, start, floor)
            # Budgets only go down from here, so one expert fewer will not fit in later rounds either
            floor = experts - 1
            log(f"round {rnd + 1}: split {split} loads with {experts} experts on the CPU; testing prompts")
            res = tuner.runtime_test(split, experts, lengths)
            free = res.get("free_mib", {})
            short = {i: free[i] for i in used_gpus if i in free and free[i] < args.margin_mib}
            if res["ok"] and not short:
                final = res
                break
            if res["oom_gpu"] is not None:
                k, cut = res["oom_gpu"], 1
            elif short:
                k = min(short, key=short.get)
                cut = max(1, math.ceil((args.margin_mib - short[k]) / 1024))
            else:
                # Failed without a recognisable out-of-memory message: treat the fullest GPU as the cause
                k = min(used_gpus, key=lambda i: free.get(i, 0))
                cut = 1
            if split[k] > g_size(gpus[k]):
                split[k] = g_size(gpus[k])
            split[k] = max(1, split[k] - cut)
            log(f"GPU {k} " + ("ran out of memory" if res["oom_gpu"] is not None else
                               f"kept only {free.get(k)} MiB free") + f"; its budget is now {split[k]} GiB")
            start = experts
    finally:
        json.dump(dict(config=str(cfg_path), gpus=gpus, trials=tuner.trials, result=final),
                  open(out_path, "w"), indent=1)

    print()
    if final is None:
        log(f"no layout passed within {args.max_rounds} rounds; see {out_path}")
        sys.exit(1)
    log("Result:")
    print(f"\n  gpu_split: [{','.join(str(x) for x in final['split'])}]")
    print(f"  cpu_moe_split_experts: {final['experts']}\n")
    print("  GPU  peak MiB  free MiB")
    for g in gpus:
        i = g["index"]
        print(f"  {i:3d}  {final['peak_mib'].get(i, 0):8d}  {final['free_mib'].get(i, 0):8d}")
    print()
    for p in final["prompts"]:
        print(f"  {p['prompt_tokens']:7d}-token prompt: prefill {p.get('prefill_tps')} T/s, "
              f"decode {p.get('decode_tps')} T/s")
    loads = sum(1 for t in tuner.trials if t["kind"] == "load")
    print(f"\n  {loads} load probes, {sum(1 for t in tuner.trials if t['kind'] == 'test')} test runs; "
          f"record in {out_path}")


def g_size(gpu: dict) -> int:
    """A card's size in whole GiB: budgets above it all mean 'the whole card'."""
    return round(gpu["total"] / 1024)


if __name__ == "__main__":
    main()
