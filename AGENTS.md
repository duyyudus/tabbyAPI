# Repository Guidelines

## Project Structure & Module Organization

TabbyAPI is a FastAPI server for ExLlamaV3 inference. `main.py` starts the server; `start.py` and `start.sh`/`start.bat` handle setup and launch. Put API routes, request schemas, and endpoint helpers under `endpoints/` (`OAI/`, `Kobold/`, and `core/`). Shared configuration, model management, and utilities live in `common/`; inference implementations live in `backends/`. Put automated tests in `tests/test_*.py`, API documentation in `docs/`, chat templates in `templates/`, and example presets in `sampler_overrides/`. `models/` and `loras/` are local asset directories, not places for source changes.

## Build, Test, and Development Commands

- `./start.sh` (or `start.bat` on Windows): create or use an environment, install the selected GPU dependencies, and launch the API.
- `python main.py`: launch directly from an already configured environment without updating dependencies.
- `pip install -e . -r tests/requirements-responses.txt`: install the project and protocol test dependencies. For GPU inference, follow `docs/01.-Getting-Started.md` for the appropriate `.[cu12]` or `.[cu13]` extra.
- `python -m pytest tests -q`: run the Python test suite.
- `ruff format --check . && ruff check .`: check formatting and linting before a pull request. Install Ruff with `pip install -e '.[dev]'`.

## Coding Style & Naming Conventions

Use Python 3.10 compatible syntax, four spaces for indentation, double quoted strings, and a 100 character line limit, as configured in `pyproject.toml`. Use `snake_case` for functions and modules, `PascalCase` for classes, and descriptive names for route and schema changes. Ruff checks production code; `tests/**` is excluded from Ruff lint rules, but keep tests readable and consistent.

## Testing Guidelines

Use pytest for automated runs; several tests use `unittest` fixtures and mocks. Name new tests `tests/test_<feature>.py` and functions `test_<behavior>`. Prefer mocked endpoint and parser tests that run without model weights or a GPU. For Responses API client compatibility, run `npm ci --prefix tests/responses_clients` followed by `python tests/responses_clients/run.py`. No coverage threshold is configured.

## Commit & Pull Request Guidelines

Recent commits commonly use a short area prefix, such as `API:`, `Tools:`, `Docs:`, `Tests:`, or `Dependencies:`, followed by an imperative summary. Use the pull request template: explain the problem, what changed and why, give an example, and link a related issue when available. Include screenshots for visible behavior and report the tests you ran.

## Configuration & Secrets

Copy `config_sample.yml` to local `config.yml` only when overriding defaults. Keep API tokens and model files out of commits; use the sample configuration files as templates.
