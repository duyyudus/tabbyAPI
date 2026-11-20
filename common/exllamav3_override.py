"""Load the exllamav3 Python package from a source checkout instead of the installed one."""

import importlib.machinery
import importlib.util
import pathlib
import sys

EXTENSION = "exllamav3_ext"


def _built_extension(root: str):
    """Spec of a compiled exllamav3_ext in the checkout root, or None (sources don't count)."""
    spec = importlib.machinery.PathFinder.find_spec(EXTENSION, [root])
    if spec and spec.origin and spec.origin.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
        return spec
    return None


class _ExllamaV3Finder:
    """
    Meta path finder that resolves the top-level `exllamav3` package from a checkout, plus the
    compiled `exllamav3_ext` extension when the checkout root holds an in-place build of it
    (`python setup.py build_ext --inplace`).

    Submodules follow the package's own __path__, so they load from the checkout too. Without an
    in-place build the extension still resolves from the installed environment. Every other
    top-level name does too, so nothing else in the checkout (tests, scripts) can shadow
    installed packages.
    """

    def __init__(self, root: pathlib.Path):
        self.root = str(root)

    def find_spec(self, name, path=None, target=None):
        if name == "exllamav3":
            return importlib.machinery.PathFinder.find_spec(name, [self.root])
        if name == EXTENSION:
            return _built_extension(self.root)
        return None


def use_exllamav3_checkout(path: str) -> pathlib.Path:
    """
    Make `import exllamav3` load from the checkout at `path` (the directory that contains the
    `exllamav3` package), and `import exllamav3_ext` too when the checkout has an in-place build.
    Returns the package directory.

    Raises ValueError when `path` holds no exllamav3 package, or RuntimeError when exllamav3 (or
    the checkout's extension) was already imported from somewhere else.
    """

    root = pathlib.Path(path).expanduser().resolve()
    package = root / "exllamav3"
    if not (package / "__init__.py").is_file():
        raise ValueError(f"no exllamav3 package found in {root}")

    loaded = sys.modules.get("exllamav3")
    if loaded is not None:
        loaded_from = pathlib.Path(loaded.__file__).resolve().parent
        if loaded_from == package:
            return package
        raise RuntimeError(f"exllamav3 was already imported from {loaded_from}")

    built = _built_extension(str(root))
    loaded_ext = sys.modules.get(EXTENSION)
    if built and loaded_ext is not None and getattr(loaded_ext, "__file__", None) != built.origin:
        raise RuntimeError(f"{EXTENSION} was already imported from {loaded_ext.__file__}")

    if not any(
        isinstance(finder, _ExllamaV3Finder) and finder.root == str(root)
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _ExllamaV3Finder(root))
    return package


def extension_origin() -> str | None:
    """File the compiled extension will load from, without importing it (None: not found)."""
    spec = importlib.util.find_spec(EXTENSION)
    return spec.origin if spec else None
