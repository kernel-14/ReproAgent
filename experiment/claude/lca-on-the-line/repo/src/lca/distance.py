#!/usr/bin/env python3
"""Lowest-common-ancestor metrics used by LCA-on-the-Line.

The paper's ID semantic error is not a generic graph distance. For two leaf
classes y_pred and y_true, N_LCA(y_pred, y_true) is the deepest shared ancestor
in the class hierarchy. Node probabilities p(y) are computed from uniform leaf
probabilities by summing descendant mass, information content is -log2 p(y),
and the LCA distance can be expressed as f(y_true) - f(N_LCA(y_true, y_pred)).
"""

from __future__ import annotations

from collections import defaultdict, deque
from math import log2
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np


Node = Any


def build_parent_index(hierarchy: Mapping[str, Any] | Sequence[Tuple[Node, Node]]) -> Dict[Node, List[Node]]:
    """Return child -> parents mapping from an edge list or hierarchy dict."""
    edges = hierarchy.get("edges", []) if isinstance(hierarchy, Mapping) else hierarchy
    parents: Dict[Node, List[Node]] = defaultdict(list)
    for child, parent in edges:
        parents[child].append(parent)
    return dict(parents)


def _children_from_parents(parents: Mapping[Node, Sequence[Node]]) -> Dict[Node, List[Node]]:
    children: Dict[Node, List[Node]] = defaultdict(list)
    for child, parent_list in parents.items():
        for parent in parent_list:
            children[parent].append(child)
    return dict(children)


def _ancestor_distances(node: Node, parents: Mapping[Node, Sequence[Node]]) -> Dict[Node, int]:
    distances: Dict[Node, int] = {node: 0}
    queue: deque[Node] = deque([node])
    while queue:
        current = queue.popleft()
        for parent in parents.get(current, []):
            next_distance = distances[current] + 1
            if parent not in distances or next_distance < distances[parent]:
                distances[parent] = next_distance
                queue.append(parent)
    return distances


def _depths_from_roots(parents: Mapping[Node, Sequence[Node]]) -> Dict[Node, int]:
    children = _children_from_parents(parents)
    nodes: Set[Node] = set(children)
    nodes.update(parents)
    for parent_list in parents.values():
        nodes.update(parent_list)
    roots = [node for node in nodes if not parents.get(node)]
    depths: Dict[Node, int] = {root: 0 for root in roots}
    queue: deque[Node] = deque(roots)
    while queue:
        current = queue.popleft()
        for child in children.get(current, []):
            next_depth = depths[current] + 1
            if child not in depths or next_depth > depths[child]:
                depths[child] = next_depth
                queue.append(child)
    return depths


def lowest_common_ancestor(class_a: Node, class_b: Node, hierarchy: Mapping[str, Any] | Sequence[Tuple[Node, Node]]) -> Optional[Node]:
    """Find the deepest common ancestor N_LCA(class_a, class_b)."""
    parents = build_parent_index(hierarchy)
    ancestors_a = _ancestor_distances(class_a, parents)
    ancestors_b = _ancestor_distances(class_b, parents)
    common = set(ancestors_a).intersection(ancestors_b)
    if not common:
        return None
    depths = hierarchy.get("depths", {}) if isinstance(hierarchy, Mapping) else {}
    if not depths:
        depths = _depths_from_roots(parents)
    return max(common, key=lambda node: (depths.get(node, 0), -ancestors_a[node] - ancestors_b[node]))


def _infer_leaves(parents: Mapping[Node, Sequence[Node]], leaves: Optional[Iterable[Node]]) -> List[Node]:
    if leaves is not None:
        return list(leaves)
    children = _children_from_parents(parents)
    nodes: Set[Node] = set(parents)
    for parent_list in parents.values():
        nodes.update(parent_list)
    return sorted([node for node in nodes if not children.get(node)], key=lambda x: str(x))


def compute_node_probabilities(
    hierarchy: Mapping[str, Any] | Sequence[Tuple[Node, Node]],
    leaves: Optional[Iterable[Node]] = None,
) -> Dict[Node, float]:
    """Compute p(node) by assigning uniform mass to leaves and summing descendants."""
    parents = build_parent_index(hierarchy)
    leaf_nodes = _infer_leaves(parents, leaves)
    if not leaf_nodes:
        return {}
    leaf_mass = 1.0 / len(leaf_nodes)
    probabilities: MutableMapping[Node, float] = defaultdict(float)
    for leaf in leaf_nodes:
        for ancestor in _ancestor_distances(leaf, parents):
            probabilities[ancestor] += leaf_mass
    return dict(probabilities)


def information_content(node: Node, node_probabilities: Mapping[Node, float], eps: float = 1e-12) -> float:
    """Return f(node) = -log2 p(node)."""
    return -log2(max(float(node_probabilities.get(node, eps)), eps))


def compute_lca_distance(
    predicted: Node,
    target: Node,
    hierarchy: Mapping[str, Any] | Sequence[Tuple[Node, Node]],
) -> float:
    """Compute hop distance from predicted/target leaves to their LCA."""
    if predicted == target:
        return 0.0
    parents = build_parent_index(hierarchy)
    pred_ancestors = _ancestor_distances(predicted, parents)
    target_ancestors = _ancestor_distances(target, parents)
    common = set(pred_ancestors).intersection(target_ancestors)
    if not common:
        return float("inf")
    lca = lowest_common_ancestor(predicted, target, hierarchy)
    return float(pred_ancestors.get(lca, 0) + target_ancestors.get(lca, 0))


def compute_lca_distance_ic(
    predicted: Node,
    target: Node,
    hierarchy: Mapping[str, Any] | Sequence[Tuple[Node, Node]],
    node_probabilities: Optional[Mapping[Node, float]] = None,
) -> float:
    """Compute D_LCA(predicted, target) = f(target) - f(N_LCA(target, predicted))."""
    if predicted == target:
        return 0.0
    probabilities = node_probabilities or compute_node_probabilities(hierarchy)
    lca = lowest_common_ancestor(target, predicted, hierarchy)
    if lca is None:
        return information_content(target, probabilities)
    return max(0.0, information_content(target, probabilities) - information_content(lca, probabilities))


def compute_lca_distance_matrix(
    hierarchy: Mapping[str, Any] | Sequence[Tuple[Node, Node]],
    class_ids: Sequence[Node],
    use_information_content: bool = False,
) -> np.ndarray:
    """Build an n x n LCA distance matrix with row i and column j as D_LCA(i, j)."""
    matrix = np.zeros((len(class_ids), len(class_ids)), dtype=float)
    probabilities = compute_node_probabilities(hierarchy, leaves=class_ids)
    for row, class_i in enumerate(class_ids):
        for col, class_j in enumerate(class_ids):
            if use_information_content:
                matrix[row, col] = compute_lca_distance_ic(class_i, class_j, hierarchy, probabilities)
            else:
                matrix[row, col] = compute_lca_distance(class_i, class_j, hierarchy)
    finite = np.isfinite(matrix)
    if not np.all(finite):
        matrix[~finite] = float(np.max(matrix[finite])) if np.any(finite) else 0.0
    return matrix


def compute_reverse_lca_matrix(lca_matrix: np.ndarray) -> np.ndarray:
    """Reverse LCA similarity used by Algorithm 1: reverse_lca = 1 - M_LCA."""
    matrix = np.asarray(lca_matrix, dtype=float)
    if matrix.size == 0:
        return matrix
    low = float(np.min(matrix))
    high = float(np.max(matrix))
    normalized = (matrix - low) / (high - low) if high > low else np.zeros_like(matrix)
    return 1.0 - normalized
