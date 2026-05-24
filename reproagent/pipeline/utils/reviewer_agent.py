"""Reviewer Agent for completion evaluation."""
import json
from langchain_core.messages import SystemMessage, HumanMessage


class ReviewerAgent:
    """Independent completion evaluator."""

    def __init__(self, llm, agent_context: dict | None = None):
        self.llm = llm
        self.agent_context = agent_context or {}

    def evaluate_completion(
        self,
        target: str,
        plan: str,
        code: str,
        execution_result: dict,
        iteration_count: int,
        *,
        preflight_result: dict | None = None,
        validation_report: dict | None = None,
        primary_metric: float | None = None,
        primary_metric_name: str = "score",
        best_round: int | None = None,
        best_metrics: dict | None = None,
    ) -> dict:
        """Evaluate task completion.

        Args:
            target: Objective
            plan: Experiment plan
            code: Generated code
            execution_result: Execution result dict
            iteration_count: Current iteration

        Returns:
            CompletionDecision dict
        """
        prompt = f"""You are a completion reviewer for ML experiment code generation.

Objective: {target}
Plan: {plan}
Current Iteration: {iteration_count}
Current Primary Metric: {primary_metric_name}={primary_metric}
Best Round So Far: {best_round}
Best Metrics: {json.dumps(best_metrics or {}, ensure_ascii=False)}

Generated Code:
```python
{code}
```

Execution Result:
- Success: {execution_result.get('success', False)}
- Output: {execution_result.get('output', '')[:500]}
- Error: {execution_result.get('error', '')[:500]}

Stage Results:
- Preflight: {json.dumps(preflight_result or {}, ensure_ascii=False)[:400]}
- Validation: {json.dumps(validation_report or {}, ensure_ascii=False)[:400]}

Decide completion status:
- "done": Objective is sufficiently achieved for this workflow stage, given current metrics and execution evidence
- "continue": Needs improvement, provide specific next action
- "blocked": Cannot proceed without user input

Output JSON:
{{
  "status": "done|continue|blocked",
  "confidence": 0.0-1.0,
  "reason": "explanation",
  "next_action": "concrete instruction for next iteration",
  "quality_score": 0.0-1.0
}}"""

        system_prompt = "You are a code completion reviewer. Output valid JSON only."

        if hasattr(self.llm, "execute"):
            response = self.llm.execute(
                f"{system_prompt}\n\n{prompt}\n\nReturn ONLY valid JSON.",
                self.agent_context,
                output_mode="text",
            )
            return self._parse_decision(response)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        response = self.llm.invoke(messages)
        return self._parse_decision(response.content)

    def _parse_decision(self, content: str) -> dict:
        """Parse LLM response to decision dict."""
        try:
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            data = json.loads(content)
            return {
                "status": data.get("status", "continue"),
                "confidence": float(data.get("confidence", 0.5)),
                "reason": data.get("reason", ""),
                "next_action": data.get("next_action", ""),
                "quality_score": float(data.get("quality_score", 0.5))
            }
        except Exception:
            return {
                "status": "continue",
                "confidence": 0.3,
                "reason": "Failed to parse reviewer response",
                "next_action": "Review code and fix issues",
                "quality_score": 0.3
            }
