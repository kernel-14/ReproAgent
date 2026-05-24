"""Agent executor abstract base class."""
from abc import ABC, abstractmethod


class AgentExecutor(ABC):
    """Abstract base class for agent executors."""

    @abstractmethod
    def execute(self, prompt: str, context: dict, output_mode: str = "code") -> str:
        """Execute agent with prompt and context.

        Args:
            prompt: The prompt to send to the agent
            context: Context dict with cwd, session_id, etc.
            output_mode: Expected response mode, e.g. "code" or "text"

        Returns:
            Agent response as string
        """
        pass
