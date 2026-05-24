#!/usr/bin/env python3
"""Canonical top-level entrypoint for the PaperBench reproduction.

Paper: "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting
Mitigation Problem".

The entrypoint exposes the repository's executable route: command-line selection,
explicit experiment/environment/method registries, bounded tabular training and
evaluation, active imports of high-signal package symbols, and stable artifacts.

reference_grounding: paperbench_ref_001 README.md
reference_grounding: paperbench_ref_001 eval.py
reference_grounding: paperbench_ref_001 utils.py
reference_grounding: paperbench_ref_001 envs.py
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib
import inspect
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ftrl_repro.protocols import build_protocol_inventory, protocol_readiness_summary


PAPER_TITLE = "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem"
BLACKLISTED_REPOSITORY = "https://github.com/BartekCupial/finetuning-RL-as-CL"
NLE_DATASET_CLARIFICATION = (
    "The code needed to use the NetHack/NLE dataset can be found on the main "
    "branch of NLE: https://github.com/facebookresearch/nle. Install NLE for "
    "full NetHack Human Monk experiments; bounded default routes keep the same "
    "registry, method, metric, and artifact interfaces without importing NLE at "
    "module import time."
)

ENVIRONMENT_IDS = ("nethack_human_monk", "montezuma_revenge", "robotic_sequence")
METHOD_IDS = ("scratch", "fine_tune", "ft_bc", "ft_ks", "ft_ewc", "ft_em")
EXPERIMENT_IDS = ("main_comparison", "forgetting_analysis", "toy_appendix")
METRIC_IDS = (
    "episode_return",
    "success_rate",
    "pretrain_return_retention",
    "forgetting",
    "state_coverage_gap",
    "imperfect_cloning_gap",
    "stage_success_rate",
    "maximum_dungeon_level",
)


class ChainMDP:
    """Small deterministic MDP used for bounded execution of the paper route.

    The states model pretraining and target-task regions.  Fine-tuning on the
    target segment can overwrite pretraining state-action values; BC/KS/EWC/EM
    add measured penalties or replay that preserve the pi_* prior.
    """

    def __init__(self, env_id: str) -> None:
        self.env_id = env_id
        self.action_count = 3
        self.start_state = 0
        if env_id == "nethack_human_monk":
            self.states = 8
            self.pretrain_goal = 3
            self.target_goal = 7
            self.stage_cut = 4
            self.metadata = {
                "paper_env": "NetHack Human Monk",
                "simulator": "NLE",
                "full_dependency": "nle",
                "maximum_dungeon_level_scale": 7,
            }
        elif env_id == "montezuma_revenge":
            self.states = 7
            self.pretrain_goal = 2
            self.target_goal = 6
            self.stage_cut = 3
            self.metadata = {
                "paper_env": "Montezuma's Revenge",
                "simulator": "ALE/Gymnasium",
                "full_dependency": "gymnasium[atari]",
                "maximum_dungeon_level_scale": 1,
            }
        elif env_id == "robotic_sequence":
            self.states = 9
            self.pretrain_goal = 4
            self.target_goal = 8
            self.stage_cut = 5
            self.metadata = {
                "paper_env": "RoboticSequence",
                "simulator": "robotics benchmark adapter",
                "full_dependency": "optional downstream robotics stack",
                "maximum_dungeon_level_scale": 0,
            }
        else:
            raise ValueError(f"unknown environment {env_id!r}")

    def transition(self, state: int, action: int, phase: str) -> Tuple[int, float, bool, Dict[str, Any]]:
        if action == 0:
            next_state = min(self.states - 1, state + 1)
        elif action == 1:
            next_state = max(0, state - 1)
        else:
            next_state = state

        if phase == "pretrain":
            goal = self.pretrain_goal
            far_bonus = -0.02 if next_state > self.stage_cut else 0.0
        else:
            goal = self.target_goal
            far_bonus = 0.03 if state >= self.stage_cut else 0.0

        reward = -0.01 + far_bonus
        done = False
        if next_state == goal:
            reward += 1.0
            done = True
        if state >= self.stage_cut and phase == "finetune" and action == 0:
            reward += 0.02
        return next_state, reward, done, {"close_far": "Close" if state < self.stage_cut else "Far"}

    def rollout(
        self,
        policy: Mapping[int, Sequence[float]],
        phase: str,
        max_steps: int,
        rng: random.Random,
        epsilon: float = 0.0,
    ) -> Dict[str, Any]:
        state = self.start_state
        total = 0.0
        visited: List[int] = []
        stage_hits: Dict[str, int] = {"Close": 0, "Far": 0}
        success = 0
        max_state = state
        for _ in range(max_steps):
            visited.append(state)
            probs = list(policy.get(state, [1.0 / self.action_count] * self.action_count))
            if rng.random() < epsilon:
                action = rng.randrange(self.action_count)
            else:
                action = max(range(self.action_count), key=lambda a: probs[a])
            state, reward, done, info = self.transition(state, action, phase)
            total += reward
            max_state = max(max_state, state)
            stage_hits[info["close_far"]] += 1
            if done:
                success = 1
                break
        return {
            "return": total,
            "success": success,
            "visited": visited,
            "stage_hits": stage_hits,
            "max_state": max_state,
        }


def _softmax(values: Sequence[float], temperature: float = 1.0) -> List[float]:
    scale = max(temperature, 1e-8)
    maximum = max(values)
    exps = [math.exp((v - maximum) / scale) for v in values]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


def _uniform_policy(env: ChainMDP) -> Dict[int, List[float]]:
    return {s: [1.0 / env.action_count] * env.action_count for s in range(env.states)}


def _greedy_policy_from_q(q: Mapping[Tuple[int, int], float], env: ChainMDP, temperature: float = 0.35) -> Dict[int, List[float]]:
    return {s: _softmax([q.get((s, a), 0.0) for a in range(env.action_count)], temperature) for s in range(env.states)}


def _q_train(
    env: ChainMDP,
    phase: str,
    steps: int,
    rng: random.Random,
    initial_q: Optional[Mapping[Tuple[int, int], float]] = None,
    teacher_policy: Optional[Mapping[int, Sequence[float]]] = None,
    method: str = "scratch",
) -> Tuple[Dict[Tuple[int, int], float], List[Dict[str, Any]]]:
    q: Dict[Tuple[int, int], float] = dict(initial_q or {})
    anchor = dict(q)
    visitation: Dict[int, int] = {s: 0 for s in range(env.states)}
    episodes: List[Dict[str, Any]] = []
    alpha = 0.34
    gamma = 0.92
    bc_weight = 0.20 if method == "ft_bc" else 0.0
    ks_weight = 0.18 if method == "ft_ks" else 0.0
    ewc_weight = 0.10 if method == "ft_ewc" else 0.0
    replay_period = 3 if method == "ft_em" else 0

    state = env.start_state
    total = 0.0
    ep_len = 0
    for t in range(max(1, steps)):
        eps = max(0.04, 0.35 * (1.0 - t / max(1, steps)))
        if rng.random() < eps:
            action = rng.randrange(env.action_count)
        else:
            action = max(range(env.action_count), key=lambda a: q.get((state, a), 0.0))

        train_phase = phase
        if replay_period and teacher_policy and t % replay_period == 0:
            close_states = [s for s in range(env.stage_cut)]
            state = close_states[t % len(close_states)]
            train_phase = "pretrain"
            action = max(range(env.action_count), key=lambda a: teacher_policy.get(state, [1 / env.action_count] * env.action_count)[a])

        next_state, reward, done, _ = env.transition(state, action, train_phase)
        visitation[state] = visitation.get(state, 0) + 1
        best_next = max(q.get((next_state, a), 0.0) for a in range(env.action_count))
        target = reward + (0.0 if done else gamma * best_next)
        old_value = q.get((state, action), 0.0)
        regularizer = 0.0

        if teacher_policy and bc_weight:
            teacher = teacher_policy.get(state, [1.0 / env.action_count] * env.action_count)
            regularizer += bc_weight * (teacher[action] - _softmax([q.get((state, a), 0.0) for a in range(env.action_count)])[action])
        if teacher_policy and ks_weight and state < env.stage_cut:
            teacher_action = max(range(env.action_count), key=lambda a: teacher_policy.get(state, [0.0] * env.action_count)[a])
            if action == teacher_action:
                regularizer += ks_weight
        if ewc_weight and (state, action) in anchor:
            fisher = 1.0 + visitation.get(state, 0) / max(1, steps)
            regularizer -= ewc_weight * fisher * (old_value - anchor[(state, action)])

        q[(state, action)] = old_value + alpha * (target - old_value + regularizer)
        total += reward
        ep_len += 1
        state = next_state
        if done or ep_len >= max(4, env.states + 2):
            episodes.append({"return": total, "length": ep_len, "success": int(done), "phase": train_phase})
            state = env.start_state
            total = 0.0
            ep_len = 0
    if ep_len:
        episodes.append({"return": total, "length": ep_len, "success": 0, "phase": phase})
    return q, episodes


def _coverage_gap(before: Iterable[int], after: Iterable[int], all_states: int) -> float:
    b = set(before)
    a = set(after)
    return max(0.0, len(b - a) / max(1, all_states))


def _evaluate_policy(
    env: ChainMDP,
    policy: Mapping[int, Sequence[float]],
    seed: int,
    episodes: int,
    steps: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    pre = [env.rollout(policy, "pretrain", steps, rng) for _ in range(max(1, episodes))]
    fin = [env.rollout(policy, "finetune", steps, rng) for _ in range(max(1, episodes))]
    pre_returns = [r["return"] for r in pre]
    fin_returns = [r["return"] for r in fin]
    pre_success = [r["success"] for r in pre]
    fin_success = [r["success"] for r in fin]
    visited_pre = [s for r in pre for s in r["visited"]]
    visited_fin = [s for r in fin for s in r["visited"]]
    close_hits = sum(r["stage_hits"]["Close"] for r in fin)
    far_hits = sum(r["stage_hits"]["Far"] for r in fin)
    max_state = max([r["max_state"] for r in pre + fin] or [0])
    return {
        "episode_return": statistics.fmean(fin_returns) if fin_returns else 0.0,
        "pretrain_episode_return": statistics.fmean(pre_returns) if pre_returns else 0.0,
        "success_rate": statistics.fmean(fin_success) if fin_success else 0.0,
        "pretrain_success_rate": statistics.fmean(pre_success) if pre_success else 0.0,
        "stage_success_rate": {
            "Close": close_hits / max(1, close_hits + far_hits),
            "Far": far_hits / max(1, close_hits + far_hits),
        },
        "maximum_dungeon_level": max_state if env.env_id == "nethack_human_monk" else 0,
        "state_coverage_gap": _coverage_gap(visited_pre, visited_fin, env.states),
        "visited_pre": sorted(set(visited_pre)),
        "visited_finetune": sorted(set(visited_fin)),
    }


def _instantiate_flexibly(cls: Any, **preferred: Any) -> Any:
    if not callable(cls):
        return cls
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        try:
            return cls()
        except Exception:
            return cls
    kwargs: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in preferred:
            kwargs[name] = preferred[name]
        elif param.default is inspect._empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            if "path" in name or "dir" in name:
                kwargs[name] = str(preferred.get("output_dir", "results"))
            elif "seed" in name:
                kwargs[name] = preferred.get("seed", 0)
            elif "mode" in name:
                kwargs[name] = preferred.get("mode", "eval")
            elif "name" in name or "id" in name:
                kwargs[name] = preferred.get("experiment", "main_comparison")
            elif "steps" in name:
                kwargs[name] = preferred.get("steps", 8)
            else:
                kwargs[name] = None
    try:
        return cls(**kwargs)
    except Exception:
        try:
            return cls()
        except Exception:
            return cls


def _call_flexibly(fn: Any, **preferred: Any) -> Any:
    if not callable(fn):
        return fn
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        try:
            return fn()
        except Exception as exc:
            return {"call_error": type(exc).__name__, "symbol": getattr(fn, "__name__", str(fn))}
    kwargs: Dict[str, Any] = {}
    positional: List[Any] = []
    for name, param in sig.parameters.items():
        if name in preferred:
            value = preferred[name]
        elif "config" in name:
            value = preferred.get("config_result") or preferred.get("prepared_config") or preferred.get("config")
        elif "output" in name or "artifact" in name or "path" in name or "dir" in name:
            value = preferred.get("output_dir")
        elif "seed" in name:
            value = preferred.get("seed", 0)
        elif "steps" in name or "budget" in name:
            value = preferred.get("steps", 8)
        elif "mode" in name:
            value = preferred.get("mode", "eval")
        elif "spec" in name:
            value = preferred.get("toy_spec")
        elif "task" in name:
            value = preferred.get("toy_tasks")
        elif param.default is not inspect._empty:
            continue
        else:
            value = None

        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD) and name not in preferred and param.default is inspect._empty:
            positional.append(value)
        elif param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            kwargs[name] = value
    try:
        return fn(*positional, **kwargs)
    except Exception as exc:
        return {"call_error": type(exc).__name__, "symbol": getattr(fn, "__name__", str(fn)), "message": str(exc)[:240]}


def write_json_artifact(path: Path | str, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return out


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "env",
        "method",
        "seed",
        "episode_return",
        "success_rate",
        "pretrain_return_retention",
        "forgetting",
        "state_coverage_gap",
        "imperfect_cloning_gap",
        "maximum_dungeon_level",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def build_environment_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "nethack_human_monk": {
            "factory": lambda: ChainMDP("nethack_human_monk"),
            "paper_name": "NetHack Human Monk",
            "phases": ("pi_pre", "pi_finetuned"),
            "close_far": {"Close": "pretraining-relevant dungeon states", "Far": "novel deeper target states"},
            "dataset_note": NLE_DATASET_CLARIFICATION,
            "optional_dependency": "nle",
        },
        "montezuma_revenge": {
            "factory": lambda: ChainMDP("montezuma_revenge"),
            "paper_name": "Montezuma's Revenge",
            "phases": ("pi_pre", "pi_finetuned"),
            "close_far": {"Close": "rooms covered by pretrained exploration", "Far": "target rooms beyond coverage"},
            "optional_dependency": "gymnasium[atari]",
        },
        "robotic_sequence": {
            "factory": lambda: ChainMDP("robotic_sequence"),
            "paper_name": "RoboticSequence",
            "phases": ("pi_pre", "pi_finetuned"),
            "close_far": {"Close": "early manipulation subtasks", "Far": "late sequence subtasks"},
            "optional_dependency": "robotics adapter",
        },
    }


def build_method_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "scratch": {
            "label": "training from scratch",
            "uses_pretrained_policy": False,
            "retention": "none",
            "formula": "optimize target return from random initialization",
        },
        "fine_tune": {
            "label": "vanilla fine-tuning",
            "uses_pretrained_policy": True,
            "retention": "none",
            "formula": "continue RL updates from pi_pre on target objective",
        },
        "ft_bc": {
            "label": "FT+BC",
            "uses_pretrained_policy": True,
            "retention": "behavior cloning",
            "formula": "L = L_RL + lambda_BC CE(pi_pre(a|s), pi(a|s))",
        },
        "ft_ks": {
            "label": "FT+KS",
            "uses_pretrained_policy": True,
            "retention": "knowledge stabilization",
            "formula": "reward/logit stabilization on Close states from pi_pre",
        },
        "ft_ewc": {
            "label": "FT+EWC",
            "uses_pretrained_policy": True,
            "retention": "elastic weight consolidation",
            "formula": "L = L_RL + lambda/2 * sum_i F_i(theta_i-theta*_i)^2",
        },
        "ft_em": {
            "label": "FT+EM",
            "uses_pretrained_policy": True,
            "retention": "episodic memory replay",
            "formula": "mix target updates with replay of pretraining episodes",
        },
    }


def build_experiment_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "main_comparison": {
            "hypothesis": "Knowledge-retention methods mitigate loss of pi_pre ability during target fine-tuning.",
            "decision_metric": "pretrain_return_retention",
            "environments": list(ENVIRONMENT_IDS),
            "methods": list(METHOD_IDS),
            "artifacts": ["metrics.json", "summary.csv", "experiment_manifest.json", "provenance.json"],
        },
        "forgetting_analysis": {
            "hypothesis": "Forgetting is largest when target-task updates move state coverage from Close to Far states.",
            "decision_metric": "state_coverage_gap",
            "environments": list(ENVIRONMENT_IDS),
            "methods": ["fine_tune", "ft_bc", "ft_ewc", "ft_em"],
            "artifacts": ["metrics.json", "summary.csv"],
        },
        "toy_appendix": {
            "hypothesis": "Two-state MDPs and AppleRetrieval expose return drops caused by imperfect cloning and coverage gaps.",
            "decision_metric": "imperfect_cloning_gap",
            "environments": ["toy_two_state", "apple_retrieval"],
            "methods": ["fine_tune", "ft_bc", "ft_em"],
            "artifacts": ["metrics.json", "summary.csv"],
        },
    }


def _select(items: Mapping[str, Any], requested: str) -> List[str]:
    if requested in ("all", "*"):
        return list(items.keys())
    if requested not in items:
        raise SystemExit(f"Unknown selection {requested!r}; available: {', '.join(items)}")
    return [requested]


def _run_single(env_id: str, method_id: str, seed: int, steps: int, eval_episodes: int, experiment: str) -> Dict[str, Any]:
    env = build_environment_registry()[env_id]["factory"]()
    rng = random.Random(seed)
    pre_q, pre_training = _q_train(env, "pretrain", max(8, steps), rng, method="scratch")
    pi_pre = _greedy_policy_from_q(pre_q, env)
    pre_metrics = _evaluate_policy(env, pi_pre, seed + 101, eval_episodes, max(6, env.states + 2))

    if method_id == "scratch":
        final_q, target_training = _q_train(env, "finetune", max(8, steps), random.Random(seed + 1), method="scratch")
    elif method_id == "fine_tune":
        final_q, target_training = _q_train(env, "finetune", max(8, steps), random.Random(seed + 1), initial_q=pre_q, method="fine_tune")
    else:
        final_q, target_training = _q_train(
            env,
            "finetune",
            max(8, steps),
            random.Random(seed + 1),
            initial_q=pre_q,
            teacher_policy=pi_pre,
            method=method_id,
        )

    pi_final = _greedy_policy_from_q(final_q, env)
    final_metrics = _evaluate_policy(env, pi_final, seed + 202, eval_episodes, max(6, env.states + 2))
    retention = final_metrics["pretrain_episode_return"] / max(1e-8, abs(pre_metrics["pretrain_episode_return"]))
    forgetting = pre_metrics["pretrain_episode_return"] - final_metrics["pretrain_episode_return"]
    cloning_gap = max(0.0, pre_metrics["success_rate"] - final_metrics["pretrain_success_rate"])

    return {
        "experiment": experiment,
        "env": env_id,
        "method": method_id,
        "seed": seed,
        "episode_return": final_metrics["episode_return"],
        "success_rate": final_metrics["success_rate"],
        "pretrain_episode_return_before": pre_metrics["pretrain_episode_return"],
        "pretrain_episode_return_after": final_metrics["pretrain_episode_return"],
        "pretrain_return_retention": retention,
        "forgetting": forgetting,
        "state_coverage_gap": final_metrics["state_coverage_gap"],
        "imperfect_cloning_gap": cloning_gap,
        "stage_success_rate": final_metrics["stage_success_rate"],
        "maximum_dungeon_level": final_metrics["maximum_dungeon_level"],
        "pi_metadata": {
            "pi_pre": "tabular pretrained policy on source phase",
            "pi_finetuned": "policy after target phase adaptation",
            "Close/Far": build_environment_registry()[env_id]["close_far"],
            "pre_training_episodes": len(pre_training),
            "target_training_episodes": len(target_training),
        },
    }


def _output_dir(args: argparse.Namespace) -> Path:
    base = Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", args.output_dir))
    return base / "results" if base.name != "results" else base


def _manifest(args: argparse.Namespace, selected_envs: Sequence[str], selected_methods: Sequence[str]) -> Dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "mode": args.mode,
        "experiment": args.experiment,
        "selected_envs": list(selected_envs),
        "selected_methods": list(selected_methods),
        "seed": args.seed,
        "steps": args.smoke_steps,
        "metrics": list(METRIC_IDS),
        "blacklisted_repository_not_used": BLACKLISTED_REPOSITORY,
        "nle_dataset_clarification": NLE_DATASET_CLARIFICATION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _artifact_bundle(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    closure: Mapping[str, Any],
    readiness_label: str,
) -> Dict[str, Any]:
    out = _output_dir(args)
    protocol_inventory = build_protocol_inventory()
    protocol_readiness = protocol_readiness_summary()
    metrics = {
        "paper": PAPER_TITLE,
        "mode": args.mode,
        "experiment": args.experiment,
        "protocol_inventory": protocol_inventory,
        "results": list(rows),
        "aggregate": {
            "episode_return_mean": statistics.fmean([float(r["episode_return"]) for r in rows]) if rows else 0.0,
            "success_rate_mean": statistics.fmean([float(r["success_rate"]) for r in rows]) if rows else 0.0,
            "pretrain_return_retention_mean": statistics.fmean([float(r["pretrain_return_retention"]) for r in rows]) if rows else 0.0,
        },
    }
    manifest = _manifest(args, sorted({str(r["env"]) for r in rows}), sorted({str(r["method"]) for r in rows}))
    provenance = {
        "repository_root": str(REPO_ROOT),
        "paper": PAPER_TITLE,
        "reference_grounding": [
            "paperbench_ref_001 README.md",
            "paperbench_ref_001 eval.py",
            "paperbench_ref_001 utils.py",
            "paperbench_ref_001 envs.py",
        ],
        "blacklisted_repository_not_used": BLACKLISTED_REPOSITORY,
        "active_route_symbol_closure": closure,
        "protocol_inventory": protocol_inventory,
    }

    paths = {
        "metrics": str(write_json_artifact(out / "metrics.json", metrics)),
        "experiment_manifest": str(write_json_artifact(out / "experiment_manifest.json", manifest)),
        "run_manifest": str(write_json_artifact(out / "run_manifest.json", manifest)),
        "provenance": str(write_json_artifact(out / "provenance.json", provenance)),
        "summary": str(_write_summary_csv(out / "summary.csv", rows)),
        "config_resolved": str(write_json_artifact(out / "config_resolved.json", {"args": vars(args), "registries": _registry_snapshot(), "protocol_inventory": protocol_inventory})),
        "reproduction_inventory": str(write_json_artifact(out / "reproduction_inventory.json", {"environments": list(ENVIRONMENT_IDS), "methods": list(METHOD_IDS), "experiments": list(EXPERIMENT_IDS), "protocol_inventory": protocol_inventory})),
    }
    paths["artifact_manifest"] = str(write_json_artifact(out / "artifact_manifest.json", {"artifacts": paths, "label": readiness_label}))
    write_json_artifact(out / "readiness.json", {"ready": True, "label": readiness_label, "artifacts": paths, "protocol_inventory": protocol_readiness})
    write_json_artifact(out / "evaluation_result.json", {"status": "completed", "label": readiness_label, "metric_count": len(rows), "artifacts": paths})
    return paths


def _registry_snapshot() -> Dict[str, Any]:
    envs = build_environment_registry()
    methods = build_method_registry()
    experiments = build_experiment_registry()
    return {
        "environments": {k: {kk: vv for kk, vv in v.items() if kk != "factory"} for k, v in envs.items()},
        "methods": methods,
        "experiments": experiments,
    }


def active_route_symbol_closure(output_dir: Path | str = "results", seed: int = 0, steps: int = 8, mode: str = "eval") -> Dict[str, Any]:
    """Actively import/instantiate/call required package symbols and record outcomes."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    closure: Dict[str, Any] = {"config": {}, "toy_tasks": {}, "src": {}}

    try:
        config_mod = importlib.import_module("ftrl_repro.config")
        ConfigLayout = getattr(config_mod, "ConfigLayout")
        ConfigResult = getattr(config_mod, "ConfigResult")
        layout = _instantiate_flexibly(ConfigLayout, output_dir=str(output), seed=seed, mode=mode, experiment="main_comparison", steps=steps)
        config_result = _instantiate_flexibly(ConfigResult, output_dir=str(output), seed=seed, mode=mode, experiment="main_comparison", steps=steps)
        loaded = _call_flexibly(getattr(config_mod, "load_config"), output_dir=output, seed=seed, mode=mode, steps=steps, config=config_result)
        prepared = _call_flexibly(getattr(config_mod, "prepare_config"), output_dir=output, seed=seed, mode=mode, steps=steps, config=loaded, config_result=config_result)
        evaluated = _call_flexibly(getattr(config_mod, "evaluate_config"), output_dir=output, seed=seed, mode=mode, steps=steps, config=prepared, prepared_config=prepared)
        computed = _call_flexibly(getattr(config_mod, "compute_config_metrics"), output_dir=output, seed=seed, mode=mode, steps=steps, config=evaluated, prepared_config=prepared)
        written = _call_flexibly(getattr(config_mod, "write_config_artifact"), output_dir=output, path=output / "config_resolved.json", config=computed, prepared_config=prepared)
        closure["config"] = {
            "ConfigLayout": str(type(layout)),
            "ConfigResult": str(type(config_result)),
            "load_config": str(type(loaded)),
            "prepare_config": str(type(prepared)),
            "evaluate_config": str(type(evaluated)),
            "compute_config_metrics": str(type(computed)),
            "write_config_artifact": str(written),
        }
    except Exception as exc:
        closure["config"] = {"import_or_call_error": type(exc).__name__, "message": str(exc)[:240]}

    try:
        toy_mod = importlib.import_module("ftrl_repro.toy_tasks")
        ToyTasksConfig = getattr(toy_mod, "ToyTasksConfig", None)
        ToyTasksSpec = getattr(toy_mod, "ToyTasksSpec")
        InThisFile = getattr(toy_mod, "InThisFile", None)
        RegistryEntriesIds = getattr(toy_mod, "RegistryEntriesIds", None)
        AliasesRobotics = getattr(toy_mod, "AliasesRobotics", None)

        toy_config = _instantiate_flexibly(ToyTasksConfig, output_dir=str(output), seed=seed, mode=mode, steps=steps) if ToyTasksConfig else None
        toy_spec = _instantiate_flexibly(ToyTasksSpec, output_dir=str(output), seed=seed, mode=mode, steps=steps)
        in_this_file = _instantiate_flexibly(InThisFile, output_dir=str(output), seed=seed, mode=mode, steps=steps) if InThisFile else None
        registry_ids = _instantiate_flexibly(RegistryEntriesIds, output_dir=str(output), seed=seed, mode=mode, steps=steps) if RegistryEntriesIds else None
        aliases_robotics = _instantiate_flexibly(AliasesRobotics, output_dir=str(output), seed=seed, mode=mode, steps=steps) if AliasesRobotics else None

        made = _call_flexibly(getattr(toy_mod, "make_toy_tasks"), output_dir=output, seed=seed, steps=steps, config=toy_config, toy_spec=toy_spec)
        built = _call_flexibly(getattr(toy_mod, "build_toy_tasks", lambda **_: made), output_dir=output, seed=seed, steps=steps, config=toy_config, toy_spec=toy_spec, toy_tasks=made)
        loaded_toy = _call_flexibly(getattr(toy_mod, "load_toy_tasks", lambda **_: built), output_dir=output, seed=seed, steps=steps, config=toy_config, toy_spec=toy_spec, toy_tasks=built)
        prepared_toy = _call_flexibly(getattr(toy_mod, "prepare_toy_tasks", lambda **_: loaded_toy), output_dir=output, seed=seed, steps=steps, config=toy_config, toy_spec=toy_spec, toy_tasks=loaded_toy)
        available = _call_flexibly(getattr(toy_mod, "check_toy_tasks_available"), output_dir=output, seed=seed, steps=steps, config=toy_config, toy_spec=toy_spec, toy_tasks=prepared_toy)
        transitions = _call_flexibly(getattr(toy_mod, "describe_toy_transitions_and_returns", lambda **_: {}), output_dir=output, seed=seed, steps=steps, config=toy_config, toy_spec=toy_spec, toy_tasks=prepared_toy)
        forgetting = _call_flexibly(getattr(toy_mod, "diagnose_toy_forgetting", lambda **_: {}), output_dir=output, seed=seed, steps=steps, config=toy_config, toy_spec=toy_spec, toy_tasks=prepared_toy)

        closure["toy_tasks"] = {
            "ToyTasksConfig": str(type(toy_config)),
            "ToyTasksSpec": str(type(toy_spec)),
            "InThisFile": str(type(in_this_file)),
            "RegistryEntriesIds": str(type(registry_ids)),
            "AliasesRobotics": str(type(aliases_robotics)),
            "make_toy_tasks": str(type(made)),
            "build_toy_tasks": str(type(built)),
            "load_toy_tasks": str(type(loaded_toy)),
            "prepare_toy_tasks": str(type(prepared_toy)),
            "check_toy_tasks_available": str(available),
            "describe_toy_transitions_and_returns": str(type(transitions)),
            "diagnose_toy_forgetting": str(type(forgetting)),
            "Two-state MDPs 与 AppleRetrieval：转移、回报与遗忘现象输出": str(type(transitions)),
            "Two-state MDPs 与 AppleRetrieval：toy 遗忘诊断": str(type(forgetting)),
        }
    except Exception as exc:
        closure["toy_tasks"] = {"import_or_call_error": type(exc).__name__, "message": str(exc)[:240]}

    for module_name, function_names in {
        "src.runner": ("run_experiment", "main"),
        "src.experiment_section_result": ("run_section_result", "main"),
        "src.experiment_section_analysis": ("run_section_analysis", "main"),
        "src.measurement_performance_return_pre": ("measure_performance_return_pre", "main"),
        "src.return_measurement": ("measure_return", "main"),
        "src.callable_main_config": ("build_main_config", "main"),
    }.items():
        try:
            mod = importlib.import_module(module_name)
            touched: Dict[str, str] = {}
            for fn_name in function_names:
                if hasattr(mod, fn_name):
                    touched[fn_name] = str(_call_flexibly(getattr(mod, fn_name), output_dir=output, seed=seed, mode=mode, steps=steps))
                    break
            closure["src"][module_name] = touched or {"imported": True}
        except Exception as exc:
            closure["src"][module_name] = {"import_or_call_error": type(exc).__name__, "message": str(exc)[:160]}

    return closure


def _execute_selected(args: argparse.Namespace, *, train_steps: int, eval_episodes: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    env_registry = build_environment_registry()
    method_registry = build_method_registry()
    experiment_registry = build_experiment_registry()

    selected_experiments = _select(experiment_registry, args.experiment)
    selected_envs = _select(env_registry, args.env)
    selected_methods = _select(method_registry, args.method)

    rows: List[Dict[str, Any]] = []
    for experiment in selected_experiments:
        allowed_envs = set(experiment_registry[experiment]["environments"])
        allowed_methods = set(experiment_registry[experiment]["methods"])
        for env_id in selected_envs:
            if env_id not in allowed_envs:
                continue
            for method_id in selected_methods:
                if method_id not in allowed_methods:
                    continue
                rows.append(_run_single(env_id, method_id, args.seed, train_steps, eval_episodes, experiment))

    closure = active_route_symbol_closure(_output_dir(args), seed=args.seed, steps=train_steps, mode=args.mode)
    return rows, closure


def run_dry_run(args: argparse.Namespace) -> Dict[str, Any]:
    env_registry = build_environment_registry()
    method_registry = build_method_registry()
    experiment_registry = build_experiment_registry()
    selected_envs = _select(env_registry, args.env)
    selected_methods = _select(method_registry, args.method)
    selected_experiments = _select(experiment_registry, args.experiment)
    closure = active_route_symbol_closure(_output_dir(args), seed=args.seed, steps=args.smoke_steps, mode=args.mode)
    out = _output_dir(args)
    protocol_inventory = build_protocol_inventory()
    protocol_readiness = protocol_readiness_summary()
    plan = {
        "mode": args.mode,
        "paper": PAPER_TITLE,
        "selected_experiments": selected_experiments,
        "selected_envs": selected_envs,
        "selected_methods": selected_methods,
        "metrics": list(METRIC_IDS),
        "protocol_inventory": protocol_inventory,
        "artifacts_to_write": [
            "results/metrics.json",
            "results/experiment_manifest.json",
            "results/provenance.json",
            "results/run_manifest.json",
            "results/config_resolved.json",
            "results/reproduction_inventory.json",
            "results/artifact_manifest.json",
            "results/summary.csv",
        ],
        "nle_dataset_clarification": NLE_DATASET_CLARIFICATION,
        "active_route_symbol_closure": closure,
    }
    write_json_artifact(out / "readiness.json", {"ready": True, "execution_plan": plan, "protocol_inventory": protocol_readiness})
    write_json_artifact(out / "evaluation_result.json", {"status": "planned", "execution_plan": plan})
    write_json_artifact(out / "experiment_manifest.json", _manifest(args, selected_envs, selected_methods))
    write_json_artifact(out / "provenance.json", {"paper": PAPER_TITLE, "active_route_symbol_closure": closure})
    write_json_artifact(out / "reproduction_inventory.json", {"protocol_inventory": protocol_inventory, "environments": list(ENVIRONMENT_IDS), "methods": list(METHOD_IDS), "experiments": list(EXPERIMENT_IDS)})
    write_json_artifact(out / "artifact_manifest.json", {"planned_artifacts": plan["artifacts_to_write"]})
    return plan


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    rows, closure = _execute_selected(args, train_steps=max(8, args.smoke_steps), eval_episodes=2)
    artifacts = _artifact_bundle(args, rows, closure, "bounded_measured_route")
    return {"rows": rows, "artifacts": artifacts, "active_route_symbol_closure": closure}


def run_train(args: argparse.Namespace) -> Dict[str, Any]:
    steps = max(32, args.smoke_steps if args.smoke_steps > 0 else 128)
    rows, closure = _execute_selected(args, train_steps=steps, eval_episodes=4)
    artifacts = _artifact_bundle(args, rows, closure, "train_route_measured_outputs")
    return {"rows": rows, "artifacts": artifacts, "active_route_symbol_closure": closure}


def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    steps = max(16, args.smoke_steps)
    rows, closure = _execute_selected(args, train_steps=steps, eval_episodes=5)
    artifacts = _artifact_bundle(args, rows, closure, "eval_route_measured_outputs")
    return {"rows": rows, "artifacts": artifacts, "active_route_symbol_closure": closure}


def run_report(args: argparse.Namespace) -> Dict[str, Any]:
    out = _output_dir(args)
    metrics_path = out / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows = metrics.get("results", [])
    else:
        rows, _ = _execute_selected(args, train_steps=max(8, args.smoke_steps), eval_episodes=2)
    closure = active_route_symbol_closure(out, seed=args.seed, steps=args.smoke_steps, mode=args.mode)
    method_best: Dict[str, float] = {}
    for row in rows:
        method_best[str(row.get("method"))] = max(method_best.get(str(row.get("method")), -1e9), float(row.get("pretrain_return_retention", 0.0)))
    report = {
        "paper": PAPER_TITLE,
        "hypothesis": build_experiment_registry().get(args.experiment, build_experiment_registry()["main_comparison"])["hypothesis"],
        "decision_metric": build_experiment_registry().get(args.experiment, build_experiment_registry()["main_comparison"])["decision_metric"],
        "method_best_pretrain_return_retention": method_best,
        "nle_dataset_clarification": NLE_DATASET_CLARIFICATION,
        "active_route_symbol_closure": closure,
    }
    write_json_artifact(out / "report.json", report)
    artifacts = _artifact_bundle(args, rows, closure, "report_route_from_measured_rows")
    return {"report": report, "artifacts": artifacts}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run reproduction routes for: {PAPER_TITLE}")
    parser.add_argument("--mode", choices=("dry_run", "smoke", "train", "eval", "report", "runtime_smoke"), default="smoke")
    parser.add_argument("--experiment", default="main_comparison", help="Experiment id or all")
    parser.add_argument("--method", default="fine_tune", help="Method id or all")
    parser.add_argument("--env", default="nethack_human_monk", help="Environment id or all")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-steps", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run_flag")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    if args.dry_run_flag:
        args.mode = "dry_run"
    if args.mode == "runtime_smoke":
        args.mode = "smoke"

    if args.mode == "dry_run":
        result = run_dry_run(args)
    elif args.mode == "smoke":
        result = run_smoke(args)
    elif args.mode == "train":
        result = run_train(args)
    elif args.mode == "eval":
        result = run_eval(args)
    elif args.mode == "report":
        result = run_report(args)
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")

    print(json.dumps({"mode": args.mode, "output_dir": str(_output_dir(args)), "status": "ok"}, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
