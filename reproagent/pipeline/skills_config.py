"""Skills configuration for reproagent module."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SkillConfig:
    """Configuration for a single skill."""
    name: str
    path: str
    enabled: bool = True
    priority: int = 0  # 0=P0, 1=P1, 2=P2


class PaperBenchReproSkillRegistry:
    """Registry for reproagent skills with validation."""

    def __init__(self, strict_validation: bool = False):
        self._skills: dict[str, SkillConfig] = {}
        self._node_bindings: dict[str, list[str]] = {}
        self._strict_validation = strict_validation

    def register_skill(self, skill: SkillConfig, nodes: list[str] | None = None):
        """Register a skill, optionally binding to nodes."""
        if skill.name in self._skills:
            raise ValueError(f"Skill '{skill.name}' already registered")

        if self._strict_validation:
            # Resolve path relative to this module's location (backend/src/modules/reproagent/)
            base_path = Path(__file__).resolve().parents[3]  # -> backend/
            skill_path = base_path / skill.path
            if not skill_path.exists():
                raise FileNotFoundError(f"Skill directory not found: {skill.path}")

        self._skills[skill.name] = skill

        if nodes:
            for node in nodes:
                self.bind_skill_to_node(node, skill.name)

    def bind_skill_to_node(self, node: str, skill_name: str):
        """Bind a registered skill to a node."""
        if skill_name not in self._skills:
            raise ValueError(f"Skill '{skill_name}' not registered")

        if node not in self._node_bindings:
            self._node_bindings[node] = []
        if skill_name not in self._node_bindings[node]:
            self._node_bindings[node].append(skill_name)

    def get_enabled_skills(self, priority: int | None = None) -> list[SkillConfig]:
        """Get enabled skills, optionally filtered by priority."""
        skills = [s for s in self._skills.values() if s.enabled]
        if priority is not None:
            skills = [s for s in skills if s.priority == priority]
        return sorted(skills, key=lambda s: s.priority)

    def get_skills_for_node(self, node_name: str) -> list[str]:
        """Get skill names bound to a node."""
        return self._node_bindings.get(node_name, []).copy()


# Default registry instance
_default_registry = PaperBenchReproSkillRegistry()

# Register existing skills
_default_registry.register_skill(
    SkillConfig(name="inno-experiment-dev", path="skills/reproagent/inno-experiment-dev", priority=0),
    nodes=["generate"]
)
_default_registry.register_skill(
    SkillConfig(name="analyze-results", path="skills/reproagent/analyze-results", priority=0),
    nodes=["evaluate"]
)
_default_registry.register_skill(
    SkillConfig(name="inno-code-survey", path="skills/reproagent/inno-code-survey", priority=0),
    nodes=["plan"]
)
_default_registry.register_skill(
    SkillConfig(name="parallel-debugging", path="skills/reproagent/parallel-debugging", priority=1, enabled=False),
    nodes=["diagnose"]
)
_default_registry.register_skill(
    SkillConfig(name="multi-reviewer-patterns", path="skills/reproagent/multi-reviewer-patterns", priority=1, enabled=False),
    nodes=["critic"]
)


class _AvailableSkillsProxy:
    """Proxy that provides live read-only view of registry."""
    def __getitem__(self, key):
        return _default_registry._skills[key]

    def __iter__(self):
        return iter(_default_registry._skills)

    def __len__(self):
        return len(_default_registry._skills)

    def keys(self):
        return _default_registry._skills.keys()

    def values(self):
        return _default_registry._skills.values()

    def items(self):
        return _default_registry._skills.items()

    def get(self, key, default=None):
        return _default_registry._skills.get(key, default)


# Compatibility layer - live view of registry
AVAILABLE_SKILLS = _AvailableSkillsProxy()


def get_enabled_skills(priority: int | None = None) -> list[SkillConfig]:
    """Get enabled skills (delegates to registry)."""
    return _default_registry.get_enabled_skills(priority)


def get_skills_dir() -> Path:
    """Get absolute path to skills directory."""
    return Path(__file__).resolve().parents[3] / "skills" / "reproagent"


def get_skills_for_node(node_name: str) -> list[str]:
    """Get skill names for a node (delegates to registry)."""
    return _default_registry.get_skills_for_node(node_name)
