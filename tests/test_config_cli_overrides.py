"""Command line values are laid over a --config file instead of being ignored."""

from common.tabby_config import TabbyConfig


def write_config(tmp_path, text):
    path = tmp_path / "config_test.yml"
    path.write_text(text)
    return str(path)


def test_cli_values_override_the_config_file(tmp_path):
    path = write_config(
        tmp_path,
        "network:\n  host: 0.0.0.0\n  port: 8001\nmodel:\n  model_name: from-file\n"
        "  max_seq_len: 4096\n",
    )
    cfg = TabbyConfig()._from_args(
        {
            "config": {"config": path},
            "network": {"port": "9000"},
            "model": {"model_name": "from-cli", "gpu_split": ["20", "24"]},
        }
    )
    assert cfg["network"] == {"host": "0.0.0.0", "port": "9000"}
    assert cfg["model"] == {
        "model_name": "from-cli",
        "max_seq_len": 4096,
        "gpu_split": ["20", "24"],
    }


def test_config_file_alone_is_used_as_is(tmp_path):
    path = write_config(tmp_path, "network:\n  port: 8001\n")
    cfg = TabbyConfig()._from_args({"config": {"config": path}, "network": {}})
    assert cfg["network"] == {"port": 8001}


def test_cli_values_without_a_config_file():
    cfg = TabbyConfig()._from_args({"config": {}, "network": {"port": "9000"}})
    assert cfg == {"network": {"port": "9000"}}
