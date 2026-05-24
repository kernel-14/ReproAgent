"""Multi-model critic executor."""
from reproagent.pipeline.config import create_node_model
from langchain_core.messages import SystemMessage, HumanMessage


class CriticExecutor:
    """Execute multi-model code review."""

    def __init__(self, model_names: list[str]):
        self.model_names = model_names

    def review(self, system_prompt: str, user_prompt: str) -> dict:
        """Execute parallel review and aggregate results."""
        reviews = []
        for model_name in self.model_names:
            model = create_node_model(model_name)
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            response = model.invoke(messages)
            reviews.append(response.content)

        # Simple consensus: all reviews agree
        consensus = len(set(reviews)) == 1
        return {
            "reviews": reviews,
            "consensus": consensus,
            "summary": reviews[0] if consensus else "Reviews differ"
        }
