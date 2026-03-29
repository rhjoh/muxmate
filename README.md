# Muxmate

A terminal-based AI assistant with custom provider support. Compatible with both OpenAI and Anthropic style messaging endpoints. Includes a single, simple bash command tool.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

Set your API key as an environment variable:

```bash
export ZAI_API_KEY="your-api-key"
```

Provider and model settings are configured in `config.py`.

## Usage

**REPL mode** (interactive):

```bash
python3 main.py --repl
```

**Single prompt**:

```bash
python3 main.py "list my files"
```

**Setup API provider**:

```bash
python3 main.py --auth
```

## Development

```bash
# Type checking
pyright main.py adapter.py agent.py config.py

# Syntax check
python3 -m py_compile main.py adapter.py agent.py config.py
```

## Versioning

This project uses [Semantic Versioning](https://semver.org/). The current version is defined in `pyproject.toml`.

- `0.x.x` — pre-stable, breaking changes may occur at any time
- `1.0.0` — first stable release

## License

MIT
