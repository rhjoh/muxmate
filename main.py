from config import load_config
import os
import subprocess
from simple_term_menu import TerminalMenu
from agent import run_repl, run_prompt
from adapter import (
    AnthropicProvider,
    OpenAIProvider,
    UserMessage,
    AssistantMessage,
)
import argparse


tools: list = [
    {
        "name": "execute_bash",
        "description": "Execute a bash command on the users machine.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["command"],
        },
    }
]


def get_tmux_pane(scrollback: int) -> str:
    pane_id = os.environ.get("TMUX_PANE")
    if pane_id:
        full_pane_history = subprocess.run(
            f"tmux capture-pane -p -J -S -{scrollback} -t {pane_id}",
            capture_output=True,
            shell=True,
            text=True,
        )
        return str(full_pane_history)
    else:
        raise RuntimeError("Not running inside a tmux pane")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", help="Setup API providers.", action="store_true")
    parser.add_argument("--repl", help="Start in REPL mode.", action="store_true")
    parser.add_argument(
        "--scroll", help="Lines of scrollback buffer to use.", type=int, default=200
    )
    parser.add_argument("prompt", nargs="?")
    # Note that this does allow "mm --repl 'list files'"
    # Add --yolo

    args = parser.parse_args()

    config = load_config()
    if config.provider_type == "anthropic":
        provider = AnthropicProvider(
            api_key=config.api_key, base_url=config.base_url, model=config.model
        )
    elif config.provider_type == "openai":
        provider = OpenAIProvider(
            api_key=config.api_key, base_url=config.base_url, model=config.model
        )
    else:
        raise ValueError(f"Unsupported provider type: {config.provider_type}")

    messages: list[UserMessage | AssistantMessage] = []
    terminal_history: str = ""

    if args.scroll and os.environ.get("TMUX_PANE"):
        terminal_history = get_tmux_pane(args.scroll)

    if args.auth:
        options = ["OpenAI", "Claude", "Kimi", "Zhipu", "Alibaba Coding"]
        terminal_menu = TerminalMenu(options)
        menu_entry_index = terminal_menu.show()
        if menu_entry_index is None:
            return
        if not isinstance(menu_entry_index, int):
            raise RuntimeError("Expected a single menu selection")
        selected_provider_key = input(f"Enter {options[menu_entry_index]} API key:\n")
        # This will eventually be saved to a user config.json or similar

    if args.repl:
        run_repl(messages, provider, config, tools, terminal_history)

    if args.prompt:
        run_prompt(
            messages=messages,
            user_prompt=args.prompt,
            provider=provider,
            config=config,
            tools=tools,
            terminal_history=terminal_history,
        )


if __name__ == "__main__":
    main()
