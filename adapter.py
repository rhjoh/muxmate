from dataclasses import dataclass, field
import json
from typing import Literal
from anthropic import Anthropic
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam


@dataclass
class ContentBlock:
    type: str  # "text | tool_use | tool_result "


@dataclass
class TextBlock(ContentBlock):
    type: str = "text"
    text: str = ""


@dataclass
class ToolUseBlock(ContentBlock):
    id: str = ""
    name: str = ""  # tool name, ex. 'execute_bash'
    command: str = ""  # ex. "ls -la""
    type: str = "tool_use"


@dataclass
class ToolResultBlock(ContentBlock):
    tool_use_id: str = ""
    name: str = ""
    output: dict = field(default_factory=dict)
    type: str = "tool_result"


@dataclass
class UserMessage:
    role: Literal["user"]
    content: list[
        ContentBlock
    ]  # would we have more than one content block in a user message?


@dataclass
class AssistantMessage:
    role: Literal["assistant"]
    content: list[ContentBlock]


class AnthropicProvider:
    def __init__(self, api_key, base_url, model):
        self.client = Anthropic(api_key=api_key, base_url=base_url)
        self.model = model

    def message_to_anthropic(self, message: UserMessage | AssistantMessage) -> dict:
        content = []  # Anthropic messages are all just role and content, so we can append directly to content here

        for block in message.content:
            if isinstance(block, TextBlock):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseBlock):
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": {"command": block.command},
                    }
                )

            elif isinstance(block, ToolResultBlock):
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": json.dumps(block.output),
                    }
                )

        return {"role": message.role, "content": content}

    def response_from_anthropic(self, response) -> AssistantMessage:
        content_blocks: list[ContentBlock] = []
        for block in response.content:
            if block.type == "text":
                content_blocks.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content_blocks.append(
                    ToolUseBlock(
                        id=block.id, name=block.name, command=block.input["command"]
                    )
                )
        return AssistantMessage(role="assistant", content=content_blocks)

    def send_message(self, system, messages, max_tokens, tools):
        anthropic_messages = []
        for message in messages:
            anthropic_messages.append(self.message_to_anthropic(message))

        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=anthropic_messages,
            max_tokens=max_tokens,
            tools=tools,
        )

        returned_message = self.response_from_anthropic(response)
        return returned_message


class OpenAIProvider:
    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def adapt_tools(self, tools) -> list[ChatCompletionToolParam]:
        if not tools:
            return []

        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    def message_to_openai(
        self, message: UserMessage | AssistantMessage
    ) -> list[dict]:  # whats the return type
        text_parts: list[str] = []
        tool_use_parts: list[dict] = []
        tool_result_parts: list[dict] = []

        for block in message.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                tool_use_parts.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps({"command": block.command}),
                        },
                    }
                )
            elif isinstance(block, ToolResultBlock):
                tool_result_parts.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": json.dumps(block.output),
                    }
                )

        if isinstance(message, UserMessage):
            if tool_result_parts:
                return tool_result_parts
            return [{"role": "user", "content": "".join(text_parts)}]

        if isinstance(message, AssistantMessage):
            openai_message = {
                "role": "assistant",
                "content": "".join(text_parts) if text_parts else None,
            }
            if tool_use_parts:
                openai_message["tool_calls"] = tool_use_parts
            return [openai_message]

    def response_from_openai(self, response) -> AssistantMessage:
        content_blocks: list[ContentBlock] = []
        message = response.choices[0].message

        if message.content:
            content_blocks.append(TextBlock(text=message.content))
        if message.tool_calls:
            for call in message.tool_calls:
                args = json.loads(call.function.arguments)
                content_blocks.append(
                    ToolUseBlock(
                        id=call.id, name=call.function.name, command=args["command"]
                    )
                )

        return AssistantMessage(role="assistant", content=content_blocks)

    def send_message(self, system, messages, max_tokens, tools):
        openai_messages = []

        if system:
            openai_messages.append({"role": "system", "content": system})
        for message in messages:
            openai_messages.extend(self.message_to_openai(message))
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_completion_tokens=max_tokens,
            tools=self.adapt_tools(tools),
        )
        returned_message = self.response_from_openai(completion)
        return returned_message
