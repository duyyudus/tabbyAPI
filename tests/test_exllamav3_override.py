"""developer.exllamav3_path loads the exllamav3 package from a source checkout."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from common.exllamav3_override import use_exllamav3_checkout

REPO = Path(__file__).resolve().parent.parent


def _make_checkout(tmp_path: Path) -> Path:
    """A fake checkout: an exllamav3 package with a submodule, plus another top-level module
    that must stay invisible (a real checkout has tests/, util/ and scripts next to it)."""
    root = tmp_path / "fork"
    (root / "exllamav3" / "sub").mkdir(parents=True)
    (root / "exllamav3" / "__init__.py").write_text("ORIGIN = 'checkout'\n")
    (root / "exllamav3" / "sub" / "__init__.py").write_text("ORIGIN = 'checkout-sub'\n")
    (root / "only_in_checkout.py").write_text("")
    return root


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_package_and_submodules_load_from_checkout(tmp_path):
    root = _make_checkout(tmp_path)
    out = _run(
        f"""
        from common.exllamav3_override import use_exllamav3_checkout
        use_exllamav3_checkout({str(root)!r})
        import exllamav3, exllamav3.sub
        print(exllamav3.ORIGIN, exllamav3.sub.ORIGIN)
        """
    )
    assert out == "checkout checkout-sub"


def test_other_top_level_names_are_not_redirected(tmp_path):
    root = _make_checkout(tmp_path)
    out = _run(
        f"""
        import importlib.util
        from common.exllamav3_override import use_exllamav3_checkout
        use_exllamav3_checkout({str(root)!r})
        print(importlib.util.find_spec("only_in_checkout") is None)
        """
    )
    assert out == "True"


def test_extension_stays_installed_without_inplace_build(tmp_path):
    root = _make_checkout(tmp_path)
    # Sources under that name are not a build
    (root / "exllamav3_ext").mkdir()
    out = _run(
        f"""
        from common.exllamav3_override import extension_origin, use_exllamav3_checkout
        use_exllamav3_checkout({str(root)!r})
        print(str(extension_origin()).startswith({str(root)!r}))
        """
    )
    assert out == "False"


def test_inplace_build_of_extension_is_used(tmp_path):
    import importlib.machinery

    root = _make_checkout(tmp_path)
    built = root / ("exllamav3_ext" + importlib.machinery.EXTENSION_SUFFIXES[0])
    built.write_bytes(b"")
    out = _run(
        f"""
        from common.exllamav3_override import extension_origin, use_exllamav3_checkout
        use_exllamav3_checkout({str(root)!r})
        print(extension_origin())
        """
    )
    assert out == str(built)


def test_user_home_is_expanded(tmp_path, monkeypatch):
    root = _make_checkout(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    # Other tests may have imported the real package in this process; isolate from that
    monkeypatch.delitem(sys.modules, "exllamav3", raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    assert use_exllamav3_checkout("~/fork") == root / "exllamav3"


def test_directory_without_package_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no exllamav3 package"):
        use_exllamav3_checkout(str(tmp_path))


def test_already_imported_elsewhere_is_rejected(tmp_path):
    root = _make_checkout(tmp_path)
    other = tmp_path / "other"
    (other / "exllamav3").mkdir(parents=True)
    (other / "exllamav3" / "__init__.py").write_text("")
    out = _run(
        f"""
        import sys
        sys.path.insert(0, {str(other)!r})
        import exllamav3
        from common.exllamav3_override import use_exllamav3_checkout
        try:
            use_exllamav3_checkout({str(root)!r})
        except RuntimeError as e:
            print("refused")
        """
    )
    assert out == "refused"
