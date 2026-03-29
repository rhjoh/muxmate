import subprocess
from adapter import (
    AssistantMessage,
    ToolResultBlock,
    TextBlock,
    ToolUseBlock,
    UserMessage,
)

SYSTEM_PROMPT = "You are an AI assistant operating inside a users terminal. You have access to tools, noteably running bash commands in the users terminal. You either terminate after a single agent turn, or run in an interactive repl mode."


def run_bash_command(bash_command: str, config) -> dict:
    tool_run_result = subprocess.run(
        bash_command,
        capture_output=True,
        shell=True,
        text=True,
        timeout=config.command_timeout_seconds,
    )
    return {
        "stdout": tool_run_result.stdout,
        "stderr": tool_run_result.stderr,
        "returncode": tool_run_result.returncode,
    }


def run_repl(messages: list, provider, config, tools: list):
    print("REPL Mode. Send 'q' to quit.")
    while True:
        user_input = input("User: ")
        if user_input == "q":
            break
        response = run_agent_turn(
            messages=messages,
            user_prompt=user_input,
            provider=provider,
            config=config,
            tools=tools,
        )
        for block in response.content:
            if isinstance(block, TextBlock):
                print(f"Agent: {block.text}")


def run_prompt(messages: list, user_prompt, provider, config, tools: list):
    response = run_agent_turn(
        messages=messages,
        user_prompt=user_prompt,
        provider=provider,
        config=config,
        tools=tools,
    )
    for block in response.content:
        if isinstance(block, TextBlock):
            print(block.text)


def run_agent_turn(
    messages: list[UserMessage | AssistantMessage],
    user_prompt: str,
    provider,
    config,
    tools: list,
):
    user_message = UserMessage(
        role="user", content=[TextBlock(text=user_prompt)]
    )  # type="text" is automatic here.
    messages.append(user_message)

    while True:
        response: AssistantMessage = provider.send_message(
            system=SYSTEM_PROMPT,
            messages=messages,
            max_tokens=config.max_tokens,
            tools=tools,
        )
        messages.append(response)

        tool_blocks = [
            tool for tool in response.content if isinstance(tool, ToolUseBlock)
        ]
        if not tool_blocks:
            break

        tool_block_results = []
        # Iterate and execute all tool calls
        for block in tool_blocks:
            command = run_bash_command(bash_command=block.command, config=config)
            print(f"⚡ Running: {block.command}")
            print(command["stdout"])
            tool_block_results.append(
                ToolResultBlock(
                    tool_use_id=block.id, name=block.name, output=command
                )  # We are adding all tool results to one final UserMessage here. Otherwise Anthropic-style endpoints seem to skip additional results.
            )
        messages.append(UserMessage(role="user", content=tool_block_results))

    return response  # Final assistant response of the turn. We only break the loop (thus ending the turn) when all tool calls are complete.
