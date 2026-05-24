"""Collaborative code generator using Claude/Codex executors."""
import logging

from .claude_sdk_wrapper import ClaudeSDKWrapper
from .codex_wrapper import CodexWrapper
from reproagent.utils.structured_output import parse_structured_output

logger = logging.getLogger(__name__)


class CollaborativeGenerator:
    """Generate code via configurable design/review and Codex implementation."""

    def __init__(self, config):
        self.mode = getattr(config, "mode", "collaborative")
        self.claude = ClaudeSDKWrapper(
            config.claude_cli_path,
            model=getattr(config, "claude_model", None),
            effort=getattr(config, "claude_effort", None),
        )
        self.codex = CodexWrapper(
            config.codex_cli_path,
            model=getattr(config, "codex_model", None),
            model_provider=getattr(config, "codex_model_provider", None),
            base_url=getattr(config, "codex_base_url", None),
            reasoning_effort=getattr(config, "codex_reasoning_effort", None),
        )
        self.max_rounds = config.max_collaboration_rounds
        if self.mode == "codex_cli_collaborative":
            self.design_executor = self.codex
            self.review_executor = self.codex
        else:
            self.design_executor = self.claude
            self.review_executor = self.claude

    def generate(self, plan: str, target: str, context: dict, iteration_context=None) -> str:
        """Generate code through collaboration.

        Args:
            plan: Experiment plan
            target: Target objective
            context: Execution context (cwd, skills_dir)
            iteration_context: Previous iteration feedback (optional)
                - previous_code: str
                - execution_result: dict
                - suggestions: list[str]
                - iteration_count: int
        """
        design = self._design(plan, target, context, iteration_context)
        code = self._codex_implement(design, target, context, iteration_context)

        for round_num in range(self.max_rounds - 1):
            review = self._review(code, design, context)
            if review.get("approved", False):
                logger.info("Round %d: Approved", round_num + 1)
                break
            logger.info("Round %d: Optimizing...", round_num + 1)
            code = self._codex_optimize(code, review.get("suggestions", []), context)

        return code

    def _execute_text(self, executor, prompt: str, context: dict) -> str:
        """Run an executor for plain text or JSON output."""
        return executor.execute(prompt, context, output_mode="text")

    def _execute_code(self, executor, prompt: str, context: dict) -> str:
        """Run an executor for raw Python code output."""
        return executor.execute(prompt, context, output_mode="code")

    def _design(self, plan: str, target: str, context: dict, iteration_context=None) -> str:
        prompt = f"""You are a senior ML experiment architect specializing in scalable data pipelines and production-grade code.

## Task
Plan: {plan}
Target: {target}"""

        if iteration_context and iteration_context.get("previous_code"):
            execution_result = dict(iteration_context.get("execution_result", {}) or {})
            iteration_count = iteration_context.get("iteration_count", 0)
            previous_code = str(iteration_context.get("previous_code", "") or "")
            prompt += f"""

## Previous Attempt (Iteration {iteration_count})
Code:
{previous_code[:500]}...

Execution Result:
- Success: {execution_result.get('success', False)}"""
            if not execution_result.get('success'):
                prompt += f"\n- Error: {str(execution_result.get('error', '') or '')[:200]}"
            if iteration_context.get('suggestions'):
                prompt += f"\n\nSuggestions:\n" + "\n".join(f"- {s}" for s in iteration_context['suggestions'][:3])

        prompt += """

## Approach
1. **Analyze First** - Understand data flow and dependencies
2. **Design for Reusability** - Modular components with clear interfaces
3. **Simple Solutions** - Avoid over-engineering
4. **Concrete Steps** - Provide actionable implementation steps

## Output Structure
1. **Analysis** - Brief assessment of the task
2. **Architecture Decision** - Key design choices (data flow, components, interfaces)
3. **Implementation Steps** - Numbered steps with pseudo-code
4. **Considerations** - Edge cases, performance, testing notes

Format: Markdown, be concise."""
        return self._execute_text(self.design_executor, prompt, context)

    def _codex_implement(self, design: str, target: str, context: dict, iteration_context=None) -> str:
        prompt = f"""You are a senior backend architect specializing in production-grade Python code.

## Design Document
{design}

## Target
{target}"""

        if iteration_context and iteration_context.get("previous_code"):
            execution_result = dict(iteration_context.get("execution_result", {}) or {})
            iteration_count = iteration_context.get("iteration_count", 0)
            prompt += f"""

## Previous Attempt (Iteration {iteration_count})
The previous implementation failed. Review and fix:
- Error: {str(execution_result.get('error', 'Unknown') or 'Unknown')[:300]}"""
            if iteration_context.get('suggestions'):
                prompt += f"\n- Suggestions: " + "; ".join(iteration_context['suggestions'][:2])

        prompt += """

## Implementation Requirements
- **Executable Python code ONLY** - No markdown, no explanations
- **Follow design exactly** - Implement all components and steps
- **Include all imports** - Complete, runnable code
- **Type hints** - Use modern Python typing
- **Clean code** - Clear naming, single responsibility

## Output Format
Pure Python code (no markdown blocks)"""
        return self._execute_code(self.codex, prompt, context)

    def _review(self, code: str, design: str, context: dict) -> dict:
        prompt = f"""You are a senior code reviewer specializing in ML code quality and best practices.

## Design
{design}

## Code to Review
{code}

## Review Checklist

### Correctness (Critical)
- [ ] Matches design specifications
- [ ] All components implemented
- [ ] Data flow is correct
- [ ] Edge cases handled

### Code Quality
- [ ] Proper error handling
- [ ] Clear naming conventions
- [ ] No code duplication
- [ ] Type hints present

### ML Best Practices
- [ ] Proper train/test split
- [ ] Random state for reproducibility
- [ ] Appropriate metrics
- [ ] Model evaluation logic

## Output Format (JSON only)
{{
  "approved": true/false,
  "issues": ["Critical issue 1", "Critical issue 2"],
  "suggestions": ["Improvement 1", "Improvement 2"]
}}"""
        result = self._execute_text(self.review_executor, prompt, context)
        try:
            return parse_structured_output(result, schema_name="collaborative_review")
        except Exception as exc:
            logger.warning("Collaborative review parsing failed: %s", exc)
            return {
                "approved": False,
                "issues": [f"Review output could not be parsed: {exc}"],
                "suggestions": ["Re-run review with valid collaborative_review JSON output."],
            }

    def _codex_optimize(self, code: str, suggestions: list, context: dict) -> str:
        prompt = f"""You are a senior backend architect. Optimize the code based on review feedback.

## Current Code
{code}

## Review Suggestions
{chr(10).join(f'- {s}' for s in suggestions)}

## Optimization Requirements
- Address ALL suggestions
- Maintain existing functionality
- Keep code clean and readable
- Preserve type hints

## Output Format
Improved Python code ONLY (no markdown blocks, no explanations)"""
        return self._execute_code(self.codex, prompt, context)
