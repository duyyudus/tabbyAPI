"""With CPU MoE offload the server process's torch thread pool follows the cores the engine leaves it."""

import torch

from backends.exllamav3 import utils


def test_cap_defaults_to_one_host_core(monkeypatch):
    monkeypatch.delenv("EXL3_MOE_HOST_CORES", raising=False)
    assert utils.cpu_moe_host_thread_cap() == 1


def test_cap_follows_host_cores_env(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_HOST_CORES", "3")
    assert utils.cpu_moe_host_thread_cap() == 3


def test_no_cap_when_host_is_unconfined(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_HOST_CORES", "0")
    assert utils.cpu_moe_host_thread_cap() is None
    monkeypatch.setattr(torch, "set_num_threads", lambda n: (_ for _ in ()).throw(AssertionError))
    assert utils.cap_host_torch_threads() is None


def test_unreadable_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_HOST_CORES", "many")
    assert utils.cpu_moe_host_thread_cap() == 1


def test_threads_lowered_and_previous_count_returned(monkeypatch):
    monkeypatch.delenv("EXL3_MOE_HOST_CORES", raising=False)
    calls = []
    monkeypatch.setattr(torch, "get_num_threads", lambda: 16)
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    assert utils.cap_host_torch_threads() == 16
    assert calls == [1]


def test_threads_never_raised(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_HOST_CORES", "4")
    calls = []
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    assert utils.cap_host_torch_threads() is None
    assert calls == []
