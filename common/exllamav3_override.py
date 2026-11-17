"""Load the exllamav3 Python package from a source checkout instead of the installed one."""

import importlib.machinery
import pathlib
import sys


class _ExllamaV3Finder:
    """
    Meta path finder that resolves only the top-level `exllamav3` package from a checkout.

    Submodules follow the package's own __path__, so they load from the checkout too. Every
    other top-level name, including the compiled `exllamav3_ext` extension, still resolves
    from the installed environment, and nothing else in the checkout (tests, scripts) can
    shadow installed packages.
    """

    def __init__(self, root: pathlib.Path):
        self.root = str(root)

    def find_spec(self, name, path=None, target=None):
        if name != "exllamav3":
            return None
        return importlib.machinery.PathFinder.find_spec(name, [self.root])


def use_exllamav3_checkout(path: str) -> pathlib.Path:
    """
    Make `import exllamav3` load from the checkout at `path` (the directory that contains the
    `exllamav3` package). Returns the package directory.

    Raises ValueError when `path` holds no exllamav3 package, or RuntimeError when exllamav3
    was already imported from somewhere else.
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

    if not any(
        isinstance(finder, _ExllamaV3Finder) and finder.root == str(root)
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _ExllamaV3Finder(root))
    return package
