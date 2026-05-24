"""Execution helpers for reproagent nodes."""
import json
from dataclasses import asdict, dataclass
from typing import Any

from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from reproagent.models.factory import create_chat_model
from reproagent.pipeline.config import create_node_model


@dataclass
class ToolCall:
    """Tracked tool call for a node execution round."""

    name: str
    args: dict[str, Any]
    id: str | None = None
    status: str = "pending"
    output: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NodeExecutor:
    """Thin model and tool execution wrapper."""

    def __init__(self, node_name: str):
        self.node_name = node_name

    def _create_model(self):
        try:
            return create_node_model(self.node_name)
        except Exception:
            return create_chat_model()

    def _bind_tools(self, model: Any, tools: list[BaseTool] | None):
        if not tools:
            return model
        binder = getattr(model, "bind_tools", None)
        if not callable(binder):
            return model
        try:
            return binder(tools)
        except Exception:
            return model

    def invoke_messages(self, messages: list[Any], tools: list[BaseTool] | None = None) -> AIMessage:
        model = self._bind_tools(self._create_model(), tools)
        response = model.invoke(messages)
        if isinstance(response, AIMessage):
            return response
        return AIMessage(content=getattr(response, "content", str(response)))

    def invoke_text(self, system: str, user: str, tools: list[BaseTool] | None = None) -> str:
        response = self.invoke_messages(
            [SystemMessage(content=system), HumanMessage(content=user)],
            tools=tools,
        )
        return self.get_text(response)

    @staticmethod
    def get_text(message: AIMessage | Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def extract_tool_calls(message: AIMessage | Any) -> list[ToolCall]:
        calls = getattr(message, "tool_calls", None) or []
        return [
            ToolCall(
                name=call.get("name", ""),
                args=call.get("args", {}) or {},
                id=call.get("id"),
            )
            for call in calls
            if call.get("name")
        ]

    @staticmethod
    def execute_tool_calls(
        tool_calls: list[ToolCall],
        tools: list[BaseTool],
    ) -> tuple[list[ToolMessage], list[ToolCall]]:
        tool_map = {tool.name: tool for tool in tools}
        messages: list[ToolMessage] = []

        for call in tool_calls:
            tool = tool_map.get(call.name)
            if tool is None:
                call.status = "error"
                call.error = f"Tool not found: {call.name}"
                messages.append(ToolMessage(content=call.error, tool_call_id=call.id or call.name))
                continue

            try:
                result = tool.invoke(call.args)
                content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                call.status = "success"
                call.output = content
            except Exception as e:
                content = f"Error: {type(e).__name__}: {e}"
                call.status = "error"
                call.error = content

            messages.append(ToolMessage(content=content, tool_call_id=call.id or call.name))

        return messages, tool_calls

    @staticmethod
    def describe_tools(tools: list[BaseTool]) -> str:
        if not tools:
            return ""
        lines = []
        for tool in tools:
            description = getattr(tool, "description", "") or ""
            lines.append(f"- **{tool.name}**: {description.splitlines()[0] if description else ''}")
        return "\n".join(lines)
