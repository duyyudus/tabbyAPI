"""prefill_priority, prefill_slice and decode_slice: validation and how they reach the engine."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backends.exllamav3 import model as model_module
from backends.exllamav3.model import ExllamaV3Container
from common.config_models import ModelConfig
from endpoints.core.types.model import ModelLoadRequest
from endpoints.core.utils.model import get_load_option_schema


def make_container(**attrs):
    container = ExllamaV3Container.__new__(ExllamaV3Container)
    container.loaded = False
    container.generator = None
    container.model = None
    container.cache = None
    container.draft_model = None
    container.draft_cache = None
    container.tokenizer = None
    container.max_batch_size = 2
    container.chunk_size = 4096
    container.draft_num_tokens = None
    container.dynamic_draft = False
    container.ngram_match_min = 0
    for key, value in attrs.items():
        setattr(container, key, value)
    return container


def create(monkeypatch, container, engine_supports=True):
    """Run create_generator against a stand-in engine; returns its kwargs and the log lines."""

    seen = {}
    logged = {"info": [], "warning": []}

    class FakeAsyncGenerator:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.error = None
            self.generator = SimpleNamespace(max_batch_size=2, recurrent_cache=None)
            if engine_supports:
                self.generator.prefill_priority = kwargs.get("prefill_priority", False)

    monkeypatch.setattr(model_module, "AsyncGenerator", FakeAsyncGenerator)
    def recorder(level):
        return lambda msg, *args, **kwargs: logged[level].append(msg)

    for level in ("info", "warning"):
        monkeypatch.setattr(model_module.xlogger, level, recorder(level))
    asyncio.run(container.create_generator())
    return seen, logged


def test_off_by_default_passes_nothing(monkeypatch):
    seen, logged = create(monkeypatch, make_container())
    assert not {"prefill_priority", "prefill_slice", "decode_slice"} & set(seen)
    assert not any("refill priority" in m for m in logged["info"] + logged["warning"])


def test_on_passes_all_three_and_says_so(monkeypatch):
    container = make_container(prefill_priority=True, prefill_slice=10.0, decode_slice=1.5)
    seen, logged = create(monkeypatch, container)
    assert seen["prefill_priority"] is True
    assert seen["prefill_slice"] == 10.0 and seen["decode_slice"] == 1.5
    assert any("a 1.5 s turn after 10 s" in m for m in logged["info"])
    assert not logged["warning"]


def test_on_without_slices_reports_no_turn(monkeypatch):
    seen, logged = create(monkeypatch, make_container(prefill_priority=True))
    assert seen["prefill_slice"] == 0.0
    assert any("no turn until prompts finish" in m for m in logged["info"])


def test_stock_engine_warns_that_it_is_ignored(monkeypatch):
    container = make_container(prefill_priority=True)
    seen, logged = create(monkeypatch, container, engine_supports=False)
    assert any("not supported by this ExLlamaV3" in m for m in logged["warning"])


def test_config_and_request_validation():
    assert ModelConfig().prefill_priority is False
    assert ModelConfig().prefill_slice == 0.0 and ModelConfig().decode_slice == 2.0
    with pytest.raises(ValidationError):
        ModelConfig(prefill_slice=-1)
    with pytest.raises(ValidationError):
        ModelConfig(decode_slice=0)
    with pytest.raises(ValidationError):
        ModelLoadRequest(model_name="m", decode_slice=0)
    request = ModelLoadRequest(model_name="m", prefill_priority=True, prefill_slice=5)
    assert request.prefill_priority and request.prefill_slice == 5


def test_options_are_in_the_load_option_schema():
    names = {
        field.name
        for group in get_load_option_schema().groups
        for field in group.fields
    }
    assert {"prefill_priority", "prefill_slice", "decode_slice"} <= names
