"""Infer paper-derived evidence obligations from paper and planning text."""

from __future__ import annotations

import re
from typing import Any


_EXPERIMENT_RE = re.compile(r"\b(?:experiment|exp\.?)\s*[-_:#]?\s*([ivxlcdm]+|\d+)\b", re.IGNORECASE)
_EXPERIMENT_RANGE_RE = re.compile(
    r"\b(?:experiment|exp\.?)\s*[-_:#]?\s*([ivxlcdm]+|\d+)\s*(?:-|–|—|to|through)\s*([ivxlcdm]+|\d+)\b",
    re.IGNORECASE,
)
_EXPERIMENT_LIST_RE = re.compile(
    r"\b(?:experiment|exp\.?)\s*[-_:#]?\s*((?:[ivxlcdm]+|\d+)(?:\s*(?:/|,|and|&)\s*(?:[ivxlcdm]+|\d+)){1,8})\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)(?:e[-+]?\d+)?", re.IGNORECASE)
_BRACED_VALUE_RE = re.compile(r"(?:\{([^{}]+)\}|\[([^\[\]]+)\]|\(([^()]+)\))")
_REFERENCE_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:references|bibliography|acknowledg(?:e)?ments?|acknowledg(?:e)?ment)\b[^\n]*"
    r"|(?:^|\n)\s*\\section\*?\s*\{\s*(?:references|bibliography|acknowledg(?:e)?ments?|acknowledg(?:e)?ment)\s*\}[^\n]*",
    re.IGNORECASE,
)
_POST_REFERENCE_MATERIAL_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:appendix|supplementary|supplemental)\b[^\n]*"
    r"|(?:^|\n)\s*\\(?:title|section)\*?\s*\{[^}]*?(?:appendix|supplementary|supplemental)[^}]*\}[^\n]*"
    r"|(?:^|\n)\s*\\appendix\b[^\n]*"
    r"|(?:^|\n)\s*\\section\*?\s*\{\s*(?:[A-Z]|[IVX]+)\.\s+(?:Detailed|Additional|Supplementary|Proof|Ablation|Implementation|Experimental)[^}]*\}[^\n]*"
    r"|(?:^|\n)\s*(?:#{1,6}\s*)?(?:[A-Z]|[IVX]+)\.\s+(?:Detailed|Additional|Supplementary|Proof|Ablation|Implementation|Experimental)\b[^\n]*",
    re.IGNORECASE,
)
_POST_REFERENCE_SOURCE_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"You do not have to|Hyperparameters?:|Paper evidence contract|paper:paper_addendum_constraints|"
    r"Binding addendum clarification|Addendum clarifications?|The addendum|You should download"
    r")\b",
    re.IGNORECASE,
)
_ROMAN_TO_INT = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
}
_INT_TO_ROMAN = {value: key for key, value in _ROMAN_TO_INT.items()}

_ENVIRONMENT_TERMS: dict[str, tuple[str, ...]] = {
    "mujoco": ("mujoco", "hopper", "walker2d", "halfcheetah", "ant-v", "humanoid", "swimmer"),
    "selfish_mining": ("selfish mining", "selfish_mining", "selfish-mining"),
    "network_defense": ("network defense", "network_defense", "network-defense", "cyber defense", "cyber_defense"),
    "autonomous_driving": ("autonomous driving", "autonomous_driving", "autonomous-driving", "driving"),
    "cage": ("cage", "cage-v", "cage challenge"),
    "atari": ("atari", "ale/", "breakout", "pong", "spaceinvaders"),
    "gym": ("openai gym", "gymnasium", "gym env", "gym environment"),
    "deepmind_control": ("deepmind control", "dmcontrol", "dmc"),
    "minigrid": ("minigrid", "mini grid"),
    "gridworld": ("gridworld", "grid world"),
    "robotics": ("robotics", "robot arm", "robotic arm", "manipulation robot"),
    "cifar": ("cifar", "cifar10", "cifar-10", "cifar100", "cifar-100"),
    "imagenet": ("imagenet", "image net"),
    "mnist": ("mnist", "fashion-mnist", "fashion mnist"),
    "wikitext": ("wikitext", "wiki text"),
    "squad": ("squad", "squad v"),
    "glue": ("glue", "superglue", "super glue"),
    "coco": ("coco", "mscoco", "ms coco", "common objects in context"),
    "laion": ("laion", "laion-400m", "laion 400m", "laion-2b", "laion 2b"),
    "clip_benchmark": ("clip benchmark", "clip_benchmark", "clip-benchmark", "robustbench"),
    "openimages": ("openimages", "open images"),
    "flickr30k": ("flickr30k", "flickr 30k"),
    "svhn": ("svhn",),
    "stl10": ("stl-10", "stl10"),
    "celeba": ("celeba",),
    "waterbirds": ("waterbirds",),
    "wilds": ("wilds",),
    "domainnet": ("domainnet", "domain net"),
}

_ABSTRACT_TERM_COVERAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "robotics": (
        "hopper",
        "walker2d",
        "reacher",
        "halfcheetah",
        "ant-v",
        "humanoid",
        "mujoco",
        "metadrive",
        "meta drive",
        "gymnasium-robotics",
        "gymnasium robotics",
        "torque",
        "end effector",
        "robot arm",
    ),
}

_DATASET_TERMS: dict[str, tuple[str, ...]] = {
    **_ENVIRONMENT_TERMS,
    "imagenet_1k": ("imagenet-1k", "imagenet 1k", "imagenet1k"),
    "imagenet_c": ("imagenet-c", "imagenet c"),
    "imagenet_r": ("imagenet-r", "imagenet r"),
    "imagenet_v2": ("imagenet-v2", "imagenet v2"),
    "imagenet_sketch": ("imagenet-sketch", "imagenet sketch"),
    "coco_2014": ("coco 2014", "mscoco 2014", "ms coco 2014"),
    "vqav2": ("vqav2", "vqa v2", "visual question answering v2"),
    "textvqa": ("textvqa", "text vqa"),
    "pope": ("pope", "pope benchmark"),
    "sqa_i": ("sqa-i", "sqa i", "science question answering"),
    "caltech101": ("caltech101", "caltech 101"),
    "stanford_cars": ("stanfordcars", "stanford cars"),
    "dtd": ("dtd", "describable textures"),
    "eurosat": ("eurosat", "euro sat"),
    "fgvc_aircraft": ("fgvc aircraft", "aircraft"),
    "flowers": ("flowers102", "flowers 102", "flowers"),
    "pcam": ("pcam", "patchcamelyon"),
    "oxford_pets": ("oxfordpets", "oxford pets"),
    "two_moons": ("two moons", "two-moons", "two_moons"),
    "gaussian_linear": ("gaussian linear", "linear gaussian"),
    "gaussian_mixture": ("gaussian mixture", "gaussian-mixture"),
    "slcp": ("slcp", "simple likelihood complex posterior"),
    "lotka_volterra": ("lotka volterra", "lotka-volterra"),
    "sird": ("sird", "susceptible infected recovered deceased"),
    "hodgkin_huxley": ("hodgkin-huxley", "hodgkin huxley"),
    "gsm8k": ("gsm8k", "grade school math"),
    "strategyqa": ("strategyqa", "strategy qa"),
    "truthfulqa": ("truthfulqa", "truthful qa"),
    "scienceqa": ("scienceqa", "science qa"),
    "toxigen": ("toxigen",),
    "ffhq": ("ffhq", "flickr-faces-hq", "flickr faces hq"),
    "lsun_church": ("lsun church", "lsun-church", "church"),
    "babies": ("babies", "10-shot babies", "10 shot babies"),
    "sunglasses": ("sunglasses", "10-shot sunglasses", "10 shot sunglasses"),
    "raphael_peale": ("raphael peale", "raphael paintings", "raphaels paintings"),
    "sketches": ("sketches", "10-shot sketches", "10 shot sketches"),
    "modigliani": ("modigliani", "amedeo modigliani", "face paintings"),
    "haunted_houses": ("haunted houses", "10-shot haunted houses", "10 shot haunted houses"),
    "landscape_drawings": ("landscape drawings", "10-shot landscape drawings", "10 shot landscape drawings"),
}

_METHOD_TERMS: dict[str, tuple[str, ...]] = {
    "ours": ("ours", "our method", "proposed method", "proposed", "rice"),
    "random": ("random baseline", "random policy baseline", "random policy"),
    "sapg": ("sapg", "split and aggregate policy gradients"),
    "statemask": ("statemask", "state mask", "state-mask"),
    "ppo": ("ppo", "proximal policy optimization"),
    "pbt": ("pbt", "dexpbt", "dex pbt", "population-based training", "population based training"),
    "pql": ("pql", "parallel q-learning", "parallel q learning"),
    "sac": ("sac", "soft actor critic", "soft actor-critic"),
    "gail": ("gail", "generative adversarial imitation"),
    "jsrl": ("jsrl", "jump-start reinforcement learning", "jumpstart reinforcement learning"),
    "bc": ("behavior cloning", "behaviour cloning", "bc"),
    "dqn": ("dqn", "deep q network", "deep q-network"),
    "ddpg": ("ddpg", "deep deterministic policy gradient"),
    "td3": ("td3", "twin delayed"),
    "trpo": ("trpo", "trust region policy optimization"),
    "cql": ("cql", "conservative q-learning", "conservative q learning"),
    "iql": ("iql", "implicit q-learning", "implicit q learning"),
    "a2c": ("a2c", "advantage actor critic"),
    "a3c": ("a3c", "asynchronous advantage actor critic"),
    "baseline": ("baseline", "baselines"),
    "chain_of_thought": ("chain-of-thought", "chain of thought", "cot prompt", "cot"),
    "oracle": ("oracle", "expert"),
    "heuristic": ("heuristic", "rule based", "rule-based"),
    "clip": ("clip model", "clip encoder", "clip features", "openai clip", "contrastive language-image pre-training", "contrastive language image pretraining"),
    "robust_clip": ("robust clip", "robust-clip"),
    "vit": ("vit", "vision transformer", "vision-transformer"),
    "resnet": ("resnet", "resnet-50", "resnet50", "resnet-101", "resnet101"),
    "bert": ("bert",),
    "roberta": ("roberta", "roberta-large", "roberta large"),
    "t5": ("t5",),
    "adapter": ("adapter tuning", "adapter-tuning", "learned adapter", "model adapter"),
    "fine_tuning": ("full fine tuning", "full fine-tuning", "supervised fine tuning", "supervised fine-tuning", "fine-tuned baseline", "finetuned baseline"),
    "lora": ("lora", "low-rank adaptation", "low rank adaptation"),
    "sft_lora": ("sft-lora", "sft lora", "supervised fine-tuning with lora", "supervised fine tuning with lora"),
    "azure_sft": ("azure-sft", "azure sft", "azure openai fine-tuning", "azure openai fine tuning"),
    "mlm": ("mlm loss", "masked language modeling", "masked word supervision"),
    "test_time_adaptation": ("test-time adaptation", "test time adaptation", "tta"),
    "prompt_tuning": ("prompt tuning", "prompt-tuning", "soft prompt"),
    "simformer": ("simformer", "all-in-one simulation-based inference"),
    "npe": ("npe", "neural posterior estimation"),
    "nle": ("nle", "neural likelihood estimation"),
    "nre": ("nre", "neural ratio estimation"),
    "bbox_adapter": ("bbox-adapter", "bbox adapter", "bboxadapter"),
    "ranking_nce": ("ranking-based nce", "ranking based nce", "noise contrastive estimation"),
    "energy_based_model": ("energy-based model", "energy based model", "ebm"),
    "online_adaptation": ("online adaptation", "online adaption"),
    "single_step_inference": ("single-step", "single step", "single-step inference", "single step inference"),
    "full_step_inference": ("full-step", "full step", "full-step inference", "full step inference"),
    "ground_truth_feedback": (
        "ground-truth feedback",
        "ground truth feedback",
        "ground-truth label feedback",
        "ground truth label feedback",
        "ground-truth solution feedback",
        "ground truth solution feedback",
    ),
    "ai_feedback": ("ai feedback", "gpt-4 feedback", "llm feedback"),
    "combined_feedback": ("combined feedback", "combined feedback setting", "combined ai and ground-truth feedback"),
    "llava": ("llava", "llava-1.5", "llava 1.5"),
    "openflamingo": ("openflamingo", "open flamingo"),
    "tecoa": ("tecoa", "tecoa^2", "tecoa^4"),
    "fare": ("fare", "fare^2", "fare^4"),
    "pgd": ("pgd", "projected gradient descent"),
    "apgd": ("apgd", "auto-pgd", "auto pgd"),
    "autoattack": ("autoattack", "auto attack"),
    "foa": ("foa", "forward-optimization adaptation", "forward optimization adaptation"),
    "lame": ("lame",),
    "t3a": ("t3a",),
    "tent": ("tent",),
    "cotta": ("cotta", "co-tuning test-time adaptation"),
    "sar": ("sar",),
    "cma_es": ("cma-es", "cma evolution strategy", "cma-based"),
    "ptq4vit": ("ptq4vit", "ptq4vit"),
    "vision_mamba": ("visionmamba", "vision mamba"),
    "diffusion_model": ("diffusion model", "diffusion models", "dpm", "dpms", "denoising diffusion"),
    "ddpm": ("ddpm", "denoising diffusion probabilistic model"),
    "ddim": ("ddim", "denoising diffusion implicit model"),
    "ldm": ("ldm", "latent diffusion model", "latent diffusion models"),
    "dpms_ant": ("dpms-ant", "dpmsant", "dpm-ant", "ddpm-ant", "ldm-ant"),
    "similarity_guided_training": ("similarity-guided training", "similarity guided training", "similarity-guided dpm", "similarity guidance"),
    "adversarial_noise_selection": ("adversarial noise selection", "adversarial noise", "worse-case noise", "worst-case noise"),
    "ddpm_pa": ("ddpm-pa", "ddpm pa", "pairwise adaptation"),
    "tgan": ("tgan",),
    "ada": ("ada", "tgan+ada", "adaptive discriminator augmentation"),
    "ewc": ("ewc", "elastic weight consolidation"),
    "cdc": ("cdc", "cross-domain consistency"),
    "dcl": ("dcl", "contrastive learning"),
}

_PARAMETER_TERMS: dict[str, tuple[str, ...]] = {
    "alpha": ("alpha", "\\alpha", "α"),
    "lambda": ("lambda", "\\lambda", "lambda_", "lam", "λ"),
    "p": ("p",),
    "population_size": ("population size k", "population_size", "population count k"),
    "prompt_count": ("number of prompt embeddings", "prompt embeddings", "N_p", "Np"),
    "source_sample_count": ("source training sample count", "number of source training samples", "source training samples"),
    "adaptation_interval": ("adaptation interval", "interval value"),
    "beta": ("beta", "\\beta", "β"),
    "gamma": ("gamma", "\\gamma", "γ"),
    "epsilon": ("epsilon", "\\epsilon", "eps", "ε"),
    "temperature": ("temperature", "temp"),
    "learning_rate": ("learning rate", "learning_rate", "lr"),
    "weight_decay": ("weight decay", "weight_decay", "wd"),
    "batch_size": ("batch size", "batch_size"),
    "patch_size": ("patch size", "patch sizes", "patch_size"),
    "epochs": ("epochs", "epoch"),
    "beam_size": ("beam size", "beam sizes", "number of beams", "beams"),
    "iteration_count": ("number of iterations", "iteration counts", "iterations", "online adaptation iterations"),
    "candidate_count": ("candidate count", "number of candidates", "candidates generated per step"),
    "adapter_size": ("adapter size", "adapter sizes", "adapter parameter count", "adapter parameter size"),
    "lora_rank": ("lora rank", "rank r", "r="),
    "shot_count": ("shot count", "number of shots", "shots", "10-shot", "10 shot", "few-shot"),
    "training_iteration_count": ("training iterations", "training iteration", "number of training iterations"),
    "similarity_guidance_scale": ("similarity guidance scale", "similarity-guided training scale"),
    "adversarial_noise_scale": ("adversarial noise scale", "omega", "\\omega", "ω"),
    "adversarial_inner_steps": ("finite-step gradient ascent", "inner maximum", "adversarial steps", "J="),
}

_METRIC_TERMS: dict[str, tuple[str, ...]] = {
    "accuracy": ("accuracy", "acc"),
    "clean_accuracy": ("clean accuracy", "standard accuracy"),
    "robust_accuracy": ("robust accuracy", "adversarial accuracy"),
    "f1": ("f1", "f1-score", "f1 score"),
    "precision": ("precision",),
    "recall": ("recall",),
    "auc": ("auc", "auroc", "roc-auc"),
    "loss": ("loss", "objective value"),
    "error_rate": ("error rate", "classification error"),
    "perplexity": ("perplexity", "ppl"),
    "bleu": ("bleu",),
    "rouge": ("rouge",),
    "fid": ("fid", "frechet inception distance", "fréchet inception distance"),
    "intra_lpips": ("intra-lpips", "intra lpips", "lpips diversity", "lpips"),
    "mse": ("mse", "mean squared error"),
    "rmse": ("rmse", "root mean squared error"),
    "mae": ("mae", "mean absolute error"),
    "reward": ("reward", "final reward", "cumulative reward"),
    "return": ("return", "episode return"),
    "fidelity_score": ("fidelity score", "fidelity"),
    "training_time": ("training time", "runtime", "wall-clock"),
    "training_cost": ("training cost", "cost of training", "train cost"),
    "inference_cost": ("inference cost", "cost of inference", "cost per token", "$/1k", "per thousand questions"),
    "api_cost": ("api cost", "api costs", "token consumption", "cost efficiency"),
    "sample_efficiency": ("sample efficiency", "sample-efficient"),
    "attack_success_rate": ("attack success rate", "asr"),
    "cider": ("cider", "cider score"),
    "vqa_accuracy": ("vqa accuracy", "visual question answering accuracy"),
    "ece": ("ece", "expected calibration error"),
    "success_rate": ("success rate", "successful attacks"),
    "memory_usage": ("memory usage", "memory consumption", "vram"),
    "gpu_memory": ("gpu memory", "gpu memory usage", "vram", "gib"),
    "c2st": ("c2st", "classifier two-sample test", "two-sample test"),
    "nll": ("nll", "negative loglikelihood", "negative log-likelihood"),
    "toxicity": ("toxicity", "toxic"),
}

_ARTIFACT_TERMS: dict[str, tuple[str, ...]] = {
    "checkpoint": ("checkpoint", "checkpoints", "model weights", "weights"),
    "trained_model": ("trained model", "pretrained model", "fine-tuned model"),
    "metrics_json": ("metrics.json", "metric artifact", "metrics artifact"),
    "result_table": ("table", "tables", "result table"),
    "result_figure": ("figure", "fig.", "figures", "plot"),
    "config": ("config", "configuration", "yaml", "json config"),
    "log": ("log", "logs", "training log"),
    "predictions": ("prediction", "predictions", "outputs"),
}
_RESULT_ARTIFACT_NAMES = {
    "checkpoint",
    "trained_model",
    "metrics_json",
    "result_table",
    "result_figure",
    "predictions",
}
_ENVIRONMENT_ONLY_DATASET_NAMES = {
    "mujoco",
    "selfish_mining",
    "network_defense",
    "autonomous_driving",
    "cage",
    "atari",
    "gym",
    "deepmind_control",
    "minigrid",
    "gridworld",
    "robotics",
}

_PARAMETER_CONTEXT_WORDS = (
    "ablation",
    "grid",
    "hyperparameter",
    "parameter",
    "sensitivity",
    "sweep",
    "value",
    "vary",
    "varied",
    "varies",
)
_METHOD_CONTEXT_WORDS = (
    "algorithm",
    "adversarial",
    "attack",
    "baseline",
    "compare",
    "comparison",
    "evaluate",
    "evaluates",
    "evaluation",
    "experiment",
    "method",
    "optimizer",
    "policy",
    "versus",
    "variant",
)
_ENVIRONMENT_CONTEXT_WORDS = (
    "application",
    "benchmark",
    "data",
    "dataset",
    "datasets",
    "domain",
    "environment",
    "env",
    "evaluate",
    "evaluation",
    "experiment",
    "scenario",
    "simulator",
    "task",
)
_POSITIVE_TERM_ACTION_RE = re.compile(
    r"\b(?:"
    r"we\s+(?:evaluate|evaluated|benchmark|benchmarked|use|used|select|selected|report|reported|compare|compared|train|trained|test|tested|run|ran)"
    r"|(?:evaluate|evaluated|evaluates|benchmark|benchmarked|benchmarks|use|used|uses|select|selected|selects|report|reported|reports|compare|compared|compares|train|trained|trains|test|tested|tests|run|runs|ran)"
    r")\b",
    re.IGNORECASE,
)
_POSITIVE_TERM_HEADING_RE = re.compile(
    r"^\s*(?:evaluation|benchmark|experiments?|results?|training|testing|comparison)\s+(?:on|with|using|for|across)\b",
    re.IGNORECASE,
)
_DATASET_CONTEXT_WORDS = (
    "benchmark",
    "classification",
    "captioning",
    "corruption",
    "data",
    "dataset",
    "datasets",
    "evaluation",
    "imagenet",
    "samples",
    "validation",
    "visual question answering",
    "vqa",
    "zero-shot",
)
_METRIC_CONTEXT_WORDS = (
    "evaluate",
    "evaluation",
    "metric",
    "metrics",
    "measure",
    "measurement",
    "report",
    "result",
    "score",
    "table",
)
_ARTIFACT_CONTEXT_WORDS = (
    "artifact",
    "export",
    "file",
    "generate",
    "output",
    "produce",
    "report",
    "result",
    "save",
    "write",
)

_METHOD_DECISION_CONTEXT_WORDS = (
    "ablation",
    "algorithm",
    "baseline",
    "baselines",
    "benchmark",
    "compare",
    "compared",
    "comparison",
    "evaluate",
    "evaluated",
    "evaluation",
    "experiment",
    "experiments",
    "figure",
    "method",
    "methods",
    "our method",
    "ours",
    "proposed method",
    "result",
    "results",
    "table",
    "variant",
    "variants",
    "we compare",
    "we test",
    "we evaluate",
)
_BACKGROUND_SECTION_MARKERS = (
    "related work",
    "preliminaries",
    "background",
)

_LATEX_TOKEN_REPLACEMENTS = {
    "\\alpha": " alpha ",
    "\\beta": " beta ",
    "\\gamma": " gamma ",
    "\\lambda": " lambda ",
    "\\epsilon": " epsilon ",
    "\\varepsilon": " epsilon ",
    "\\eta": " eta ",
    "\\theta": " theta ",
    "\\tau": " tau ",
    "\\omega": " omega ",
    "\\in": " in ",
    "\\times": " x ",
    "\\%": "%",
    "α": " alpha ",
    "β": " beta ",
    "γ": " gamma ",
    "λ": " lambda ",
    "ε": " epsilon ",
    "ω": " omega ",
}


def _dedupe(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _normalized_text(value: str) -> str:
    lowered = _latex_to_plain_text(str(value or "")).lower()
    lowered = lowered.replace("_", " ").replace("-", " ")
    lowered = re.sub(r"[^a-z0-9.+/]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _latex_to_plain_text(value: str) -> str:
    text = str(value or "")
    for source, target in _LATEX_TOKEN_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = text.replace("\\{", "{").replace("\\}", "}")
    text = text.replace("\\[", "[").replace("\\]", "]")
    text = re.sub(r"\$+", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = text.replace("∈", " in ")
    text = text.replace("≤", "<=").replace("≥", ">=")
    return text


def _compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _strip_reference_sections(text: str) -> str:
    """Remove reference/acknowledgement spans without discarding later appendices."""
    source = str(text or "")
    pieces: list[str] = []
    cursor = 0
    for match in _REFERENCE_HEADING_RE.finditer(source):
        start = match.start()
        if start < cursor:
            continue
        pieces.append(source[cursor:start])
        continuation = _POST_REFERENCE_MATERIAL_RE.search(source, match.end())
        if continuation:
            cursor = continuation.start()
            continue
        post_reference_source = _POST_REFERENCE_SOURCE_MARKER_RE.search(source, match.end())
        cursor = post_reference_source.start() if post_reference_source else len(source)
    pieces.append(source[cursor:])
    return "\n".join(part for part in pieces if part.strip())


def _strip_background_method_sections(text: str) -> str:
    """Remove literature/background sections before deriving experiment obligations."""
    source = str(text or "")
    lines = source.splitlines()
    kept: list[str] = []
    skipping = False
    for line in lines:
        normalized = _normalized_text(line)
        looks_like_heading = (
            bool(normalized)
            and len(normalized.split()) <= 8
            and (
                line.lstrip().startswith("#")
                or line.lstrip().startswith("\\section")
                or line.lstrip().startswith("\\subsection")
                or normalized[:2].isdigit()
            )
        )
        if looks_like_heading and any(marker in normalized for marker in ("related work", "preliminaries", "background")):
            skipping = True
            continue
        if skipping and looks_like_heading:
            skipping = False
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


def _contains_alias(text: str, alias: str) -> bool:
    if not alias:
        return False
    lowered = str(text or "").lower()
    alias_lower = str(alias or "").lower()
    alias_variants = [alias_lower]
    figure_match = re.search(r"\b(fig(?:ure)?\.?)\s*([0-9]+[a-z]?)\b", alias_lower)
    if figure_match:
        number = figure_match.group(2)
        alias_variants.extend(
            [
                f"fig {number}",
                f"fig. {number}",
                f"figure {number}",
                f"figure_{number}",
                f"fig_{number}",
            ]
        )
    table_match = re.search(r"\btable\s*([0-9]+[a-z]?)\b", alias_lower)
    if table_match:
        number = table_match.group(1)
        alias_variants.extend([f"table {number}", f"table_{number}"])
    for alias_candidate in _dedupe(alias_variants):
        alias_boundary = r"(?<![a-z0-9])" + re.escape(alias_candidate).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        if re.search(alias_boundary, lowered):
            return True
    normalized = _normalized_text(lowered)
    for alias_candidate in _dedupe(alias_variants):
        alias_normalized = _normalized_text(alias_candidate)
        if not alias_normalized:
            continue
        normalized_boundary = r"(?<![a-z0-9])" + re.escape(alias_normalized).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        if re.search(normalized_boundary, normalized):
            return True
    # Compact matching covers spelling variants such as StateMask/state mask, but
    # is unsafe for short aliases like p, lr, bc, or eps because they occur inside
    # ordinary words.
    compacted_text = _compact_text(lowered)
    for alias_candidate in _dedupe(alias_variants):
        alias_compact = _compact_text(alias_candidate)
        if len(alias_compact) >= 5 and alias_compact in compacted_text:
            return True
    return False


def _has_any_alias(text: str, aliases: tuple[str, ...] | list[str]) -> bool:
    return any(_contains_alias(text, alias) for alias in aliases)


def _window_has_context(text: str, term_aliases: tuple[str, ...], context_words: tuple[str, ...], *, window: int = 80) -> bool:
    lowered = str(text or "").lower()
    for alias in term_aliases:
        alias_lower = str(alias or "").lower()
        if not alias_lower:
            continue
        for match in re.finditer(re.escape(alias_lower), lowered):
            start = max(0, match.start() - window)
            end = min(len(lowered), match.end() + window)
            snippet = lowered[start:end]
            if any(word in snippet for word in context_words):
                return True
    return False


def _canonical_experiment_token(raw: str) -> tuple[str, list[str]]:
    value = str(raw or "").strip().lower()
    if value.isdigit():
        number = int(value)
        roman = _INT_TO_ROMAN.get(number, value)
    else:
        roman = value
        number = _ROMAN_TO_INT.get(roman)
    canonical = f"experiment_{roman}"
    aliases = [
        f"experiment {roman}",
        f"experiment_{roman}",
        f"experiment-{roman}",
        f"exp {roman}",
        f"exp_{roman}",
        f"exp-{roman}",
        f"experiment{roman}",
        f"exp{roman}",
    ]
    if number is not None:
        aliases.extend(
            [
                f"experiment {number}",
                f"experiment_{number}",
                f"experiment-{number}",
                f"exp {number}",
                f"exp_{number}",
                f"exp-{number}",
            ]
        )
    return canonical, _dedupe(aliases)


def _experiment_number(raw: str) -> int | None:
    value = str(raw or "").strip().lower()
    if value.isdigit():
        return int(value)
    return _ROMAN_TO_INT.get(value)


def _experiment_raw_tokens_from_list(raw: str) -> list[str]:
    return [
        token.strip()
        for token in re.split(r"\s*(?:/|,|and|&)\s*", str(raw or ""), flags=re.IGNORECASE)
        if token.strip()
    ]


def _extract_named_experiments(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    for match in _EXPERIMENT_RANGE_RE.finditer(str(text or "")):
        start = _experiment_number(match.group(1))
        end = _experiment_number(match.group(2))
        if start is None or end is None or start > end or end - start > 20:
            continue
        for number in range(start, end + 1):
            raw = _INT_TO_ROMAN.get(number, str(number))
            canonical, aliases = _canonical_experiment_token(raw)
            if canonical in seen:
                continue
            seen.add(canonical)
            items.append({"name": canonical, "aliases": aliases})

    for match in _EXPERIMENT_LIST_RE.finditer(str(text or "")):
        for raw in _experiment_raw_tokens_from_list(match.group(1)):
            canonical, aliases = _canonical_experiment_token(raw)
            if canonical in seen:
                continue
            seen.add(canonical)
            items.append({"name": canonical, "aliases": aliases})

    for match in _EXPERIMENT_RE.finditer(str(text or "")):
        canonical, aliases = _canonical_experiment_token(match.group(1))
        if canonical in seen:
            continue
        seen.add(canonical)
        items.append({"name": canonical, "aliases": aliases})
    return items


def _extract_term_group(text: str, terms: dict[str, tuple[str, ...]], *, context_words: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    normalized_text = _latex_to_plain_text(text)
    for name, aliases in terms.items():
        if not _has_any_alias(normalized_text, aliases):
            continue
        if context_words and name not in {"ours", "baseline"} and not _window_has_context(normalized_text, aliases, context_words):
            continue
        if name == "recall" and not _has_metric_recall_context(normalized_text):
            continue
        items.append({"name": name, "aliases": list(_dedupe([name, *aliases]))})
    return items


def _has_metric_recall_context(text: str) -> bool:
    lowered = _latex_to_plain_text(str(text or "")).lower()
    metric_recall_patterns = (
        r"\b(?:precision|f1|f1-score|specificity|sensitivity)\s*(?:[,/&]|\band\b|\bor\b)\s*recall\b",
        r"\brecall\s*(?:[,/&]|\band\b|\bor\b)\s*(?:precision|f1|f1-score|specificity|sensitivity)\b",
        r"\brecall\s+(?:score|metric|rate|at\s+k|@k)\b",
        r"\b(?:score|metric|metrics|measured|measure|reported|reporting|evaluate|evaluation)\b[^.\n]{0,80}\brecall\b",
        r"\brecall\b[^.\n]{0,80}\b(?:score|metric|metrics|measured|measure|reported|reporting|evaluate|evaluation)\b",
    )
    return any(re.search(pattern, lowered) for pattern in metric_recall_patterns)


def _sentence_windows(text: str, *, include_markers: tuple[str, ...], exclude_markers: tuple[str, ...] = ()) -> str:
    """Return sentences likely to define experiment obligations instead of background mentions."""
    source = _latex_to_plain_text(str(text or ""))
    pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", source) if piece.strip()]
    selected: list[str] = []
    for index, sentence in enumerate(pieces):
        lowered = sentence.lower()
        if exclude_markers and any(marker in lowered for marker in exclude_markers):
            continue
        if not any(marker in lowered for marker in include_markers):
            continue
        start = max(0, index - 1)
        end = min(len(pieces), index + 2)
        selected.extend(pieces[start:end])
    return "\n".join(_dedupe(selected))


def _sentences_with_term_actions(
    text: str,
    terms: dict[str, tuple[str, ...]],
) -> str:
    """Return sentences that positively bind known terms to experiment actions."""
    source = _latex_to_plain_text(str(text or ""))
    sentences = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", source) if piece.strip()]
    selected: list[str] = []
    for sentence in sentences:
        if not (_POSITIVE_TERM_ACTION_RE.search(sentence) or _POSITIVE_TERM_HEADING_RE.search(sentence)):
            continue
        if any(_has_any_alias(sentence, aliases) for aliases in terms.values()):
            selected.append(sentence)
    return "\n".join(_dedupe(selected))


def _extract_method_group(text: str) -> list[dict[str, Any]]:
    decision_text = _sentence_windows(
        text,
        include_markers=_METHOD_DECISION_CONTEXT_WORDS,
        exclude_markers=_BACKGROUND_SECTION_MARKERS,
    )
    source = decision_text or text
    methods = _extract_term_group(source, _METHOD_TERMS, context_words=_METHOD_CONTEXT_WORDS)
    if decision_text:
        background_methods = {
            str(item.get("name", "") or "")
            for item in _extract_term_group(text, _METHOD_TERMS, context_words=_METHOD_CONTEXT_WORDS)
        }
        decision_methods = {str(item.get("name", "") or "") for item in methods}
        # Generic "baseline" is useful only when no concrete compared baselines were found.
        if "baseline" in decision_methods and len(decision_methods - {"baseline", "ours"}) >= 1:
            methods = [item for item in methods if item.get("name") != "baseline"]
        # Preserve proposed-method aliases from the full paper when the decision window names only "ours".
        if "ours" in background_methods and "ours" not in decision_methods:
            methods.append({"name": "ours", "aliases": list(_dedupe(["ours", *_METHOD_TERMS["ours"]]))})
    method_names = {str(item.get("name", "") or "") for item in methods}
    if "combined_feedback" in method_names:
        combined_windows = _sentence_windows(
            text,
            include_markers=("combined ai and ground-truth feedback", "combined ai and ground truth feedback", "combined feedback"),
        ).lower()
        if combined_windows and "ground_truth_feedback" in method_names and "ground truth feedback setting" not in combined_windows:
            methods = [item for item in methods if item.get("name") != "ground_truth_feedback"]
    return _dedupe_dicts(methods)


def _contains_other_parameter_alias(text: str, current_name: str) -> bool:
    return any(
        name != current_name and _has_any_alias(text, aliases)
        for name, aliases in _PARAMETER_TERMS.items()
    )


def _sentence_fragment_after(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+|[;\n]", str(text or ""), maxsplit=1)[0]


def _sentence_fragment_before(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+|[;\n]", str(text or ""))[-1]


def _has_similarity_guidance_context(text: str) -> bool:
    lowered = str(text or "").lower().replace("_", " ").replace("-", " ")
    return any(
        token in lowered
        for token in (
            "similarity guidance",
            "similarity guided",
            "similarity-guided",
            "dpms ant",
            "dpm ant",
            "guidance scale",
        )
    )


def _has_population_size_control_context(text: str) -> bool:
    lowered = str(text or "").lower().replace("_", " ").replace("-", " ")
    return any(
        token in lowered
        for token in (
            "population based training",
            "population-based training",
            "population size k",
            "population_size",
            "pbt",
            "dexpbt",
            "evolution strategy",
            "evolutionary search",
        )
    )


def _parameter_value_aliases(text: str, name: str, aliases: tuple[str, ...]) -> list[str]:
    normalized = re.sub(r"\s+", " ", _latex_to_plain_text(text))
    alias_pattern = "|".join(re.escape(_latex_to_plain_text(alias).strip()) for alias in aliases if _latex_to_plain_text(alias).strip())
    if not alias_pattern:
        return []
    values: list[str] = []
    source_text = _latex_to_plain_text(text)

    def table_values_between(start_pattern: str, end_pattern: str, allowed: set[str] | None = None) -> list[str]:
        for start_match in re.finditer(start_pattern, source_text, flags=re.IGNORECASE):
            remaining = source_text[start_match.end():]
            end_match = re.search(end_pattern, remaining, flags=re.IGNORECASE | re.DOTALL)
            block_end = start_match.end() + end_match.start() if end_match else len(source_text)
            block = source_text[start_match.start():block_end]
            rows = re.findall(r"(?m)^\s*([0-9]+(?:\.[0-9]+)?)\s*&", block)
            if allowed is not None:
                rows = [value for value in rows if value in allowed]
            rows = _dedupe(rows)
            if len(rows) >= 2:
                return rows
        return []


    if name == "beam_size":
        for match in re.finditer(r"\bk\s*=\s*([0-9,\s]+)", source_text, flags=re.IGNORECASE):
            direct = _NUMBER_RE.findall(match.group(1))
            if len(direct) >= 2:
                return _dedupe(direct)
    if name == "iteration_count":
        direct_values: list[str] = []
        for match in re.finditer(r"\bT\s*=\s*([0-9,\s]+)", source_text):
            direct_values.extend(_NUMBER_RE.findall(match.group(1)))
        direct_values = _dedupe(direct_values)
        if len(direct_values) >= 2:
            return direct_values
    if name == "adapter_size":
        direct = re.findall(r"\b(0\.[13])\s*B\b", source_text, flags=re.IGNORECASE)
        if len(set(direct)) >= 2:
            return _dedupe(direct)
    if name == "similarity_guidance_scale":
        if not _has_similarity_guidance_context(source_text):
            return []
        direct = table_values_between(r"(?:^|\n)\s*(?:table\s+\d+\.[^\n]*gamma|effects of\s+.*?gamma)", r"(?:^|\n)\s*(?:table\s+\d+\.|effects of adversarial|effects of training|gpu memory)")
        if len(direct) >= 2:
            return direct[:8]
    if name == "adversarial_noise_scale":
        direct = table_values_between(r"(?:^|\n)\s*(?:table\s+\d+\.[^\n]*omega|effects of\s+adversarial noise)", r"(?:^|\n)\s*(?:table\s+\d+\.|effects of training|gpu memory)")
        if len(direct) >= 2:
            return direct[:8]
    if name == "training_iteration_count":
        direct = table_values_between(r"(?:^|\n)\s*(?:table\s+\d+\.[^\n]*iteration|effects of training iteration)", r"(?:^|\n)\s*(?:gpu memory|table\s+\d+\.)", {"0", "50", "100", "150", "200", "250", "300", "350", "400", "500", "5000"})
        if len(direct) >= 2:
            return direct[:8]

    braced_value = r"(?:\{([^{}]+)\}|\[([^\[\]]+)\]|\(([^()]+)\))"
    direct_patterns = (
        rf"^\s*(?:(?:are\s+)?(?:=|:|in|from|values?|value)|are\s+in|are)\s*(?:the\s*)?(?:space|set|range|grid)?\s*{braced_value}",
        rf"^\s*(?:for\s+(?:{alias_pattern})\s*)?(?:in|from)\s*(?:the\s*)?(?:space|set|range|grid)?\s*{braced_value}",
        rf"^\s*[^.;\n]{{0,70}}?\b(?:in|from|values?|value)\b\s*(?:are\s*)?(?:the\s*)?(?:space|set|range|grid)?\s*{braced_value}",
    )
    sweep_context_words = (
        "analysis",
        "configured",
        "different",
        "experiment",
        "experiments",
        "grid",
        "hyper-parameter",
        "hyperparameter",
        "impact",
        "parameter choice",
        "plotting",
        "sensitivity",
        "space",
        "sweep",
        "vary",
        "varied",
        "varies",
        "values",
    )
    for match in re.finditer(rf"(?<![a-z0-9_])(?:{alias_pattern})(?![a-z0-9_])", normalized, flags=re.IGNORECASE):
        local_context = normalized[max(0, match.start() - 180): min(len(normalized), match.end() + 220)].lower()
        if not any(word in local_context for word in sweep_context_words):
            continue
        after = normalized[match.end() :]
        before = normalized[max(0, match.start() - 220): match.start()]
        sentence_after = _sentence_fragment_after(after)[:220]
        sentence_before = _sentence_fragment_before(before)[-220:]
        for pattern in direct_patterns:
            direct = re.search(pattern, sentence_after, flags=re.IGNORECASE)
            if not direct:
                continue
            value_group_start = next(
                (direct.start(index) for index, group in enumerate(direct.groups(), start=1) if group),
                -1,
            )
            if value_group_start > 0 and _contains_other_parameter_alias(sentence_after[:value_group_start], name):
                continue
            direct_text = " ".join(group for group in direct.groups() if group)
            direct_values = _NUMBER_RE.findall(direct_text)
            if len(direct_values) >= 2:
                values.extend(direct_values)
                break
        if values:
            continue
        local_sentence = " ".join([sentence_before, sentence_after])
        local_numbers = _NUMBER_RE.findall(local_sentence)
        if name == "beam_size":
            beam_values = [value for value in local_numbers if value in {"1", "3", "5"}]
            if len(set(beam_values)) >= 2:
                values.extend(beam_values)
        elif name == "iteration_count":
            iteration_context = any(
                token in local_sentence.lower()
                for token in ("t=", "t =", "number of iterations", "different numbers of iterations", "online adaptation")
            )
            if not iteration_context or any(token in local_sentence.lower() for token in ("beam size", "number of beams", "beams")):
                continue
            t_match = re.search(r"\bT\s*=\s*([0-9,\s]+)", local_sentence)
            if t_match:
                values.extend(_NUMBER_RE.findall(t_match.group(1)))
                continue
            iteration_values = [value for value in local_numbers if value in {"0", "1", "2", "3", "4", "5"}]
            if len(set(iteration_values)) >= 3:
                values.extend(iteration_values)
        elif name == "adapter_size":
            adapter_values = [value for value in local_numbers if value in {"0.1", "0.3", "0.1b", "0.3b"}]
            if len(set(adapter_values)) >= 2:
                values.extend(adapter_values)
        elif name == "shot_count":
            shot_values = [value for value in local_numbers if value in {"10", "100"}]
            if len(set(shot_values)) >= 1 and ("shot" in local_sentence.lower() or "few-shot" in local_sentence.lower()):
                values.extend(shot_values)
        elif name == "training_iteration_count":
            if "training" not in local_sentence.lower() and "iteration" not in local_sentence.lower():
                continue
            iteration_values = [value for value in local_numbers if value in {"150", "300", "400", "500", "5000", "5,000"}]
            if len(set(iteration_values)) >= 2:
                values.extend(iteration_values)
        elif name == "patch_size":
            patch_values = [value for value in local_numbers if value in {"1", "2", "4", "8", "16", "32", "64"}]
            if len(set(patch_values)) >= 2:
                values.extend(patch_values)
        elif name == "similarity_guidance_scale":
            if not _has_similarity_guidance_context(local_sentence):
                continue
            gamma_values = [value for value in local_numbers if value in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}]
            if len(set(gamma_values)) >= 2 or "gamma=5" in local_sentence.lower() or "gamma = 5" in local_sentence.lower():
                values.extend(gamma_values or ["5"])
        elif name == "adversarial_noise_scale":
            omega_values = [value for value in local_numbers if value in {"0.01", "0.02", "0.03", "0.05", "0.1", "0.2", "0.3"}]
            if len(set(omega_values)) >= 2 or "omega=0.02" in local_sentence.lower() or "omega = 0.02" in local_sentence.lower():
                values.extend(omega_values or ["0.02"])
    return _dedupe(values)


def _strong_sweep_parameter_names(text: str) -> set[str]:
    normalized = re.sub(r"\s+", " ", _latex_to_plain_text(str(text or "")).lower())
    sweep_markers = (
        "ablation",
        "configured",
        "grid",
        "grid search",
        "analysis",
        "different",
        "impact",
        "parameter setting",
        "sensitivity",
        "setting",
        "settings",
        "sweep",
        "sweep over",
        "swept",
        "vary",
        "varied",
        "varies",
        "varying",
        "we set",
        "we vary",
    )
    relation_markers = (
        "for",
        "on",
        "over",
        "with",
        "with respect to",
        "w.r.t",
    )
    names: set[str] = set()
    for sentence in re.split(r"[.;\n]", normalized):
        if not any(marker in sentence for marker in sweep_markers):
            continue
        if not any(marker in sentence for marker in relation_markers):
            continue
        sentence_names = {
            name
            for name, aliases in _PARAMETER_TERMS.items()
            if _has_any_alias(sentence, aliases)
        }
        if len(sentence_names) >= 1:
            names.update(sentence_names)
    return names


def _parameter_has_explicit_experiment_control_context(text: str, name: str, aliases: tuple[str, ...]) -> bool:
    """Return whether a parameter mention is an experiment control, not only notation."""
    normalized = re.sub(r"\s+", " ", _latex_to_plain_text(str(text or "")).lower())
    alias_pattern = "|".join(
        re.escape(_latex_to_plain_text(alias).strip())
        for alias in aliases
        if _latex_to_plain_text(alias).strip()
    )
    if not alias_pattern:
        return False
    generic_control_markers = (
        "ablation",
        "configured",
        "grid search",
        "hyperparameter",
        "parameter setting",
        "sensitivity",
        "setting",
        "settings",
        "sweep",
        "sweep over",
        "swept",
        "vary",
        "varied",
        "varies",
        "varying",
        "values",
        "we set",
        "we use",
        "we vary",
    )
    positive_control_markers = (
        "ablation",
        "configured",
        "grid search",
        "hyperparameter",
        "parameter setting",
        "sensitivity",
        "setting",
        "settings",
        "sweep over",
        "swept",
        "vary",
        "varied",
        "varies",
        "varying",
        "values",
        "we set",
        "we use",
        "we vary",
    )
    narrow_symbol_markers = (
        "ablation",
        "grid",
        "sensitivity",
        "sweep",
        "vary",
        "varied",
        "varies",
        "varying",
    )
    attack_markers = ("attack", "apgd", "pgd", "perturbation", "robust")
    for match in re.finditer(rf"(?<![a-z0-9_])(?:{alias_pattern})(?![a-z0-9_])", normalized, flags=re.IGNORECASE):
        local = normalized[max(0, match.start() - 180): min(len(normalized), match.end() + 220)]
        if name in {"alpha", "beta", "gamma", "epsilon", "p"}:
            if any(marker in local for marker in narrow_symbol_markers):
                return True
            if name in {"alpha", "epsilon", "gamma"} and any(marker in local for marker in attack_markers):
                return True
            if re.search(rf"(?<![a-z0-9_]){re.escape(name)}\s*(?:=|:|in\b|values?\b)", local):
                return True
            continue
        if "learning rate" in local and name in {"lambda", "learning_rate"}:
            return True
        if name == "temperature" and not any(marker in local for marker in positive_control_markers):
            continue
        if any(marker in local for marker in generic_control_markers):
            return True
    return False


def _extract_parameter_sweeps(text: str) -> list[dict[str, Any]]:
    lowered = _latex_to_plain_text(str(text or "")).lower()
    strong_sweep_names = _strong_sweep_parameter_names(lowered)
    items: list[dict[str, Any]] = []
    for name, aliases in _PARAMETER_TERMS.items():
        if name == "p":
            has_param = bool(
                re.search(r"\bp\s*(?:=|:|in\b|values?\b|sweep\b|grid\b)", lowered)
                or re.search(r"\b(?:parameter|probability|sensitivity)\s+p\b", lowered)
            )
        else:
            has_param = _has_any_alias(lowered, aliases)
        if not has_param:
            continue
        value_aliases = _parameter_value_aliases(lowered, name, aliases)
        if name in {"population_size", "lora_rank"} and len(value_aliases) < 2:
            continue
        if name == "population_size" and not _has_population_size_control_context(lowered):
            continue
        has_sweep_context = _window_has_context(
            lowered,
            aliases,
            ("analysis", "configured", "different", "experiment", "grid", "impact", "sensitivity", "sweep", "vary", "varied", "varies", "values", "space"),
            window=160,
        )
        allow_single_value = name == "shot_count" and bool(value_aliases)
        if len(value_aliases) < 2 and not has_sweep_context and not allow_single_value:
            continue
        if len(value_aliases) < 2 and name not in strong_sweep_names and not allow_single_value:
            continue
        if len(value_aliases) < 2 and not allow_single_value and not _parameter_has_explicit_experiment_control_context(
            lowered, name, aliases
        ):
            continue
        if (
            name in {"alpha", "beta", "gamma", "epsilon", "p"}
            and len(value_aliases) < 2
            and not _parameter_has_explicit_experiment_control_context(lowered, name, aliases)
        ):
            continue
        if name in {"alpha", "beta", "gamma", "epsilon"} and len(value_aliases) < 2 and name in strong_sweep_names:
            formula_context = any(
                token in lowered
                for token in (
                    "differential equation",
                    "differential equations",
                    "ode",
                    "recovery rate",
                    "mortality rate",
                    "contact rate",
                    "population sizes",
                    "prey",
                    "predator",
                    "susceptible",
                    "infected",
                )
            )
            if formula_context:
                continue
        if name == "iteration_count" and set(value_aliases).issubset({"0", "1"}):
            continue
        if (
            len(value_aliases) < 2
            and name in {"source_sample_count", "candidate_count", "alpha", "epsilon", "gamma", "iteration_count"}
            and name not in strong_sweep_names
        ):
            continue
        if len(value_aliases) < 2 and name in {"training_iteration_count", "similarity_guidance_scale", "adversarial_noise_scale", "adversarial_inner_steps"}:
            continue
        items.append({"name": name, "aliases": list(_dedupe([name, *aliases])), "values": value_aliases[:8]})
    return items


def _extract_result_artifacts(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in re.finditer(r"\b(?:Table|Figure|Fig\.?)\s+\d+[A-Za-z]?\b", str(text or ""), flags=re.IGNORECASE):
        name = match.group(0).strip().replace("Fig.", "Figure")
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append({"name": name, "aliases": [name]})
    for item in _extract_term_group(text, _ARTIFACT_TERMS, context_words=_ARTIFACT_CONTEXT_WORDS):
        key = str(item.get("name", "")).lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    return items



def _item(name: str, aliases: list[str] | tuple[str, ...]) -> dict[str, Any]:
    return {"name": name, "aliases": list(_dedupe([name, *list(aliases)]))}


def _extract_protocol_obligations(text: str) -> list[dict[str, Any]]:
    normalized = _normalized_text(text)
    lowered = _latex_to_plain_text(str(text or "")).lower()
    obligations: list[dict[str, Any]] = []

    if "half precision" in normalized or "half precision" in lowered or "fp16" in lowered:
        obligations.append(_item("half_precision_attack", ["half precision", "fp16", "float16"]))
    if "single precision" in normalized or "single precision" in lowered or "fp32" in lowered:
        obligations.append(_item("single_precision_attack", ["single precision", "fp32", "float32"]))
    if (
        "lowest" in normalized
        and any(metric in normalized for metric in ("cider", "vqa accuracy", "score"))
        and any(term in normalized for term in ("ground truth", "ground truths", "caption", "answer"))
    ):
        obligations.append(
            _item(
                "per_sample_lowest_score_selection",
                [
                    "lowest cider score",
                    "lowest score across all ground truth",
                    "lowest score across all ground-truth",
                    "per sample",
                    "stored",
                    "selected and stored",
                ],
            )
        )
    if "target string" in normalized and any(term in normalized for term in ("contained exactly", "exactly contained", "contains exactly")):
        obligations.append(_item("exact_target_string_success", ["target string", "contained exactly", "success"] ))
    if (
        any(term in normalized for term in ("randomly sampled", "randomly sample", "random sample"))
        and any(term in normalized for term in ("coco", "dataset", "images", "samples"))
    ):
        obligations.append(_item("random_sample_manifest", ["randomly sampled", "random sample", "coco", "image ids", "manifest"]))
    if any(term in normalized for term in ("stock image", "stock images", "handpicked", "hand picked")):
        obligations.append(_item("stock_image_manifest", ["stock images", "handpicked", "hand picked", "source manifest"]))
    if any(term in normalized for term in ("after each attack", "tracked throughout", "calculated after each")):
        obligations.append(_item("per_attack_metric_tracking", ["after each attack", "tracked throughout", "calculated after each"]))
    if any(term in normalized for term in ("transfer attack", "transfer attacks", "adversarial examples")):
        obligations.append(_item("transfer_attack_evaluation", ["transfer attack", "adversarial examples", "evaluate on adversarial examples"]))
    if "pope" in normalized:
        obligations.append(_item("pope_benchmark_protocol", ["pope", "yes/no", "precision", "recall", "f1"]))
    if any(term in normalized for term in ("sqa i", "sqa-i", "science question answering")):
        obligations.append(_item("sqa_i_benchmark_protocol", ["sqa-i", "science question answering", "evaluation parsing"]))
    if any(term in normalized for term in ("jailbreak", "jailbreaking")):
        obligations.append(_item("jailbreak_attack_protocol", ["jailbreak", "harmful prompts", "single image", "5000 iterations"]))
    return _dedupe_dicts(obligations)


def _dedupe_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        name = str(item.get("name", "") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(item)
    return ordered


def _extract_fixed_hyperparameters(text: str) -> list[dict[str, Any]]:
    source = _latex_to_plain_text(str(text or ""))
    lowered = source.lower()
    normalized = _normalized_text(source)
    items: list[dict[str, Any]] = []

    def add(name: str, aliases: list[str]) -> None:
        items.append(_item(name, aliases))

    if re.search(r"\b100\s+iterations?\b", lowered):
        add("100_iterations", ["100 iterations", "iterations=100", "num_iterations=100"])
    if re.search(r"\b10000\s+iterations?\b", lowered) or re.search(r"\b10,000\s+iterations?\b", lowered):
        add("10000_iterations", ["10000 iterations", "10,000 iterations", "iterations=10000"])
    if re.search(r"\b5000\s+iterations?\b", lowered) or re.search(r"\b5,000\s+iterations?\b", lowered):
        add("5000_iterations", ["5000 iterations", "5,000 iterations", "iterations=5000"])
    if re.search(r"\b300\s+(?:training\s+)?iterations?\b", lowered):
        add("300_training_iterations", ["300 training iterations", "300 iterations", "train_iterations=300"])
    if re.search(r"\b10\s*[- ]shot\b", lowered) or re.search(r"\b10\s+training images?\b", lowered):
        add("10_shot_setting", ["10-shot", "10 shot", "10 training images", "few-shot"])
    if re.search(r"\bgamma\s*=\s*5\b", lowered):
        add("gamma_5", ["gamma=5", "gamma = 5", "similarity guidance scale 5"])
    if re.search(r"\bomega\s*=\s*0\.02\b", lowered):
        add("omega_0.02", ["omega=0.02", "omega = 0.02", "adversarial noise scale 0.02"])
    if re.search(r"\bj\s*=\s*10\b", lowered):
        add("adversarial_inner_steps_10", ["J=10", "J = 10", "10 adversarial steps", "finite-step gradient ascent"])
    if re.search(r"\b25\s+(?:randomly\s+sampled\s+)?(?:coco\s+)?images?\b", lowered) or re.search(r"\b25\s+(?:handpicked|hand-picked|stock)\b", lowered):
        add("25_samples", ["25 images", "25 samples", "25 randomly sampled", "25 handpicked", "25 hand-picked", "25 stock", "sample_size=25", "n=25"])
    if re.search(r"\bfive\s+ground[- ]truth\b", lowered) or re.search(r"\b5\s+ground[- ]truth\b", lowered):
        add("5_ground_truths", ["five ground-truth", "five ground truth", "5 ground-truth", "ground_truths=5"])
    if re.search(r"\b2\s+epochs?\b", lowered) or "two epochs" in lowered:
        add("2_epochs", ["2 epochs", "two epochs", "epochs=2"])
    if re.search(r"\b10\s+(?:steps?\s+of\s+)?pgd\b", lowered) or re.search(r"\bpgd\s+steps?\s*(?:=|:)?\s*10\b", lowered):
        add("10_pgd_steps", ["10 PGD steps", "pgd_steps=10"])
    if "4/255" in lowered or "4 / 255" in lowered:
        add("epsilon_4/255", ["4/255", "4 / 255", "epsilon=4/255", "epsilon 4/255"])
    if "2/255" in lowered or "2 / 255" in lowered:
        add("epsilon_2/255", ["2/255", "2 / 255", "epsilon=2/255", "epsilon 2/255"])
    if "1/255" in lowered or "1 / 255" in lowered:
        add("alpha_1/255", ["1/255", "1 / 255", "alpha=1/255", "step size alpha = 1/255"])
    if "adamw" in normalized and ("0.9" in normalized or "0 9" in normalized) and ("0.95" in normalized or "0 95" in normalized):
        add("adamw_betas_0.9_0.95", ["beta_1 = 0.9", "beta_2 = 0.95", "betas=(0.9, 0.95)", "betas 0.9 0.95"])
    if "weight decay" in normalized and ("1e 4" in normalized or "1e-4" in lowered or "0 0001" in normalized):
        add("weight_decay_1e-4", ["weight decay 1e-4", "weight_decay=1e-4", "0.0001"])
    if re.search(r"\bbatch\s+size\b[^.\n;]{0,40}\b128\b", lowered) or re.search(r"\bbatch_size\s*=\s*128\b", lowered):
        add("batch_size_128", ["batch size 128", "batch_size=128"])
    if re.search(r"\bbatch\s+size\b[^.\n;]{0,40}\b64\b", lowered) or re.search(r"\bbatch_size\s*=\s*64\b", lowered):
        add("batch_size_64", ["batch size 64", "batch_size=64"])
    if re.search(r"\bbatch\s+size\b[^.\n;]{0,40}\b32\b", lowered) or re.search(r"\bbatch_size\s*=\s*32\b", lowered):
        add("batch_size_32", ["batch size 32", "batch_size=32"])
    if re.search(r"\b(?:3|three)\s+(?:random\s+)?seeds?\b", lowered) or re.search(r"\bseeds?\s*(?:=|:)\s*\[?\s*(?:0|1)[,\s]+(?:1|2)[,\s]+(?:2|3)", lowered):
        add("three_seed_protocol", ["3 seeds", "three seeds", "random seeds", "mean and std"])
    if re.search(r"\b64\s+(?:tiles?|blocks?)\b", lowered):
        add("mask_tiles_64", ["64 tiles", "64 blocks", "mask_tiles=64"])
    if re.search(r"\bp\s*=\s*0\.3\b", lowered) or re.search(r"\bprobability\s+(?:of\s+)?0\.3\b", lowered):
        add("mask_probability_0.3", ["p=0.3", "probability 0.3", "mask probability 0.3"])
    if "nearest neighbor" in normalized or "nearest-neighbor" in lowered:
        add("nearest_neighbor_upsample", ["nearest neighbor", "nearest-neighbor", "nearest_neighbor"])
    if "momentum" in normalized and "0 9" in normalized:
        add("momentum_0.9", ["momentum 0.9", "momentum=0.9"])
    if "learning rate" in normalized and "0 001" in normalized:
        add("learning_rate_0.001", ["learning rate 0.001", "lr=0.001"])
    if "learning rate" in normalized and "0 05" in normalized:
        add("learning_rate_0.05", ["learning rate 0.05", "lr=0.05"])
    if any(term in normalized for term in ("cosine decay", "cosine schedule")) and "warmup" in normalized:
        add("cosine_decay_with_linear_warmup", ["cosine decay", "linear warmup", "cosine decay schedule with linear warmup"])
    items = _dedupe_dicts(items)
    batch_size_matches = re.findall(
        r"\bbatch\s+size\b[^.\n;:]{0,80}\b(128|64|32)\b|\bbatch_size\s*=\s*(128|64|32)\b",
        lowered,
    )
    explicit_batch_sizes = {left or right for left, right in batch_size_matches}
    if len(explicit_batch_sizes) > 1:
        priority = ["32", "64", "128"]
        selected = next((value for value in priority if value in explicit_batch_sizes), "")
        if selected:
            items = [
                item
                for item in items
                if not str(item.get("name", "") or "").startswith("batch_size_")
                or str(item.get("name", "") or "") == f"batch_size_{selected}"
            ]
    return items


def _extract_trend_obligations(text: str) -> list[dict[str, Any]]:
    lowered = _latex_to_plain_text(str(text or "")).lower()
    normalized = _normalized_text(lowered)
    normalized_sentences = [
        _normalized_text(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+|[;\n]+", lowered)
        if sentence.strip()
    ]
    obligations: list[dict[str, Any]] = []
    has_p0 = bool(re.search(r"\bp\s*(?:=|==)\s*0(?:\.0+)?\b", lowered))
    has_p1 = bool(re.search(r"\bp\s*(?:=|==)\s*1(?:\.0+)?\b", lowered))
    has_endpoint_pair = bool(
        re.search(r"endpoints?[^.\n]{0,80}\bp\b|\bp\b[^.\n]{0,80}endpoints?", lowered)
    )
    has_low_language = any(word in lowered for word in ("lowest", "minimum", "minima", "worst", "least", "low at endpoints"))
    endpoint_low = ((has_p0 and has_p1) or has_endpoint_pair) and has_low_language
    if not endpoint_low:
        endpoint_low = bool(
            re.search(r"performance\s+is\s+low\s+when\s+p\s*=\s*0", lowered)
            and re.search(r"or\s+p\s*=\s*1", lowered)
        )
    if endpoint_low:
        obligations.append(
            {
                "name": "endpoint_low",
                "aliases": ["endpoint", "boundary", "p=0", "p = 0", "p==0", "p=1", "p = 1", "p==1", "lowest", "minimum", "worst"],
            }
        )
    insensitive = (
        any(word in normalized for word in ("not sensitive", "not that sensitive", "insensitive", "does not change much", "does not vary a lot", "stable across"))
        and any(param in normalized for param in ("alpha", "lambda", "learning rate", "temperature"))
    )
    if insensitive:
        obligations.append(
            {
                "name": "sweep_insensitive",
                "aliases": ["not sensitive", "insensitive", "stable", "robust", "sensitivity"],
            }
        )
    outperformance = any(word in normalized for word in ("outperform", "outperforms", "better than", "improves over", "improvement over"))
    if outperformance and any(word in normalized for word in ("baseline", "baselines", "random", "state mask", "statemask")):
        obligations.append(
            {
                "name": "baseline_outperformance",
                "aliases": ["outperform", "better than", "improve", "baseline", "comparison"],
            }
        )
    improves_when_positive = False
    for sentence in normalized_sentences:
        has_parameter = any(param in sentence for param in ("lambda", "alpha", "temperature", "learning rate"))
        has_positive_value = any(word in sentence for word in ("greater than 0", "> 0", "positive", "nonzero", "non zero"))
        has_improvement = any(word in sentence for word in ("improve", "improves", "higher", "better"))
        if has_parameter and has_positive_value and has_improvement:
            improves_when_positive = True
            break
    if improves_when_positive:
        obligations.append(
            {
                "name": "positive_parameter_improves",
                "aliases": ["greater than 0", "> 0", "positive", "nonzero", "improves", "higher"],
            }
        )
    return obligations


def _extract_implementation_obligations(
    text: str,
    *,
    datasets: list[dict[str, Any]],
    environments: list[dict[str, Any]],
    methods: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    parameter_sweeps: list[dict[str, Any]],
    fixed_hyperparameters: list[dict[str, Any]],
    protocol_obligations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Infer code-path obligations that cannot be satisfied by prose alone."""
    lowered = _latex_to_plain_text(str(text or "")).lower()
    normalized = _normalized_text(lowered)
    method_names = {str(item.get("name", "") or "") for item in methods}
    protocol_names = {str(item.get("name", "") or "") for item in protocol_obligations}
    obligations: list[dict[str, Any]] = []

    def add(name: str, aliases: list[str]) -> None:
        obligations.append(_item(name, aliases))

    environment_names = {str(item.get("name", "") or "") for item in environments}
    non_environment_datasets = [
        item
        for item in datasets
        if str(item.get("name", "") or "") not in _ENVIRONMENT_ONLY_DATASET_NAMES
        or str(item.get("name", "") or "") not in environment_names
    ]
    if non_environment_datasets:
        add(
            "dataset_prepare_validate_path",
            [
                "dataset registry",
                "prepare_dataset",
                "download_dataset",
                "validate_dataset",
                "availability check",
                "data manifest",
                "make_dataset",
                "data_loader",
            ],
        )
    if len(datasets) >= 2 or len(environments) >= 2:
        add(
            "benchmark_registry_matrix",
            [
                "benchmark registry",
                "dataset registry",
                "environment registry",
                "protocol matrix",
                "experiment matrix",
                "experiment_",
                "task registry",
            ],
        )
    model_like_methods = {
        "clip",
        "robust_clip",
        "vit",
        "resnet",
        "bert",
        "roberta",
        "t5",
        "llava",
        "openflamingo",
        "vision_mamba",
    }
    if method_names.intersection(model_like_methods) or any(term in normalized for term in ("pretrained", "checkpoint", "model weights")):
        add(
            "model_loader_factory_path",
            [
                "load_model",
                "model_factory",
                "create_model",
                "from_pretrained",
                "load_state_dict",
                "checkpoint",
                "pretrained",
                "weights",
            ],
        )
    if metrics:
        add(
            "metric_formula_aggregation_path",
            [
                "compute_metric",
                "metric_fn",
                "aggregate_metrics",
                "accuracy",
                "f1",
                "cider",
                "ece",
                "success_rate",
                "robust_accuracy",
                "metric",
            ],
        )
    result_artifact_names = {str(item.get("name", "") or "") for item in artifacts}.intersection(_RESULT_ARTIFACT_NAMES)
    if result_artifact_names:
        add(
            "artifact_writer_path",
            [
                "write_artifact",
                "artifact_writer",
                "json.dump",
                "write_json",
                "to_csv",
                "savefig",
                "table_writer",
                "report_writer",
            ],
        )
    if parameter_sweeps or fixed_hyperparameters:
        add(
            "hyperparameter_config_path",
            [
                "hyperparameter",
                "config",
                "dataclass",
                "yaml",
                "argparse",
                "sweep",
                "schedule",
                "optimizer",
                "parameter_sweep",
            ],
        )
    if protocol_obligations:
        add(
            "per_sample_protocol_bookkeeping_path",
            [
                "per_sample",
                "sample_manifest",
                "track",
                "record",
                "bookkeeping",
                "manifest",
                "selected",
                "stored",
                "store",
            ],
        )
    if (
        method_names.intersection({"pgd", "apgd", "autoattack"})
        or any("attack" in name for name in protocol_names)
        or any(term in normalized for term in ("adversarial", "attack", "jailbreak"))
    ):
        add(
            "attack_or_adaptation_algorithm_path",
            [
                "attack",
                "adapt",
                "pgd",
                "apgd",
                "autoattack",
                "epsilon",
                "iterations",
                "step_size",
                "objective",
            ],
        )
    if any(term in normalized for term in ("train", "training", "fine tuning", "fine-tuning", "pretrain", "pre-training", "epoch")):
        add(
            "training_or_finetuning_loop_path",
            [
                "train",
                "training_loop",
                "fit",
                "optimizer",
                "epoch",
                "epochs",
                "loss",
                "backward",
                "scheduler",
            ],
        )
    if metrics or datasets or environments or any(term in normalized for term in ("evaluate", "evaluation", "benchmark")):
        add(
            "evaluation_loop_path",
            [
                "evaluate",
                "run_evaluation",
                "benchmark",
                "dataloader",
                "inference",
                "prediction",
                "aggregate",
                "experiment_",
            ],
        )
    return _dedupe_dicts(obligations)


def infer_evidence_contract(text: str) -> dict[str, Any]:
    """Infer a compact evidence contract from paper/addendum or planning text."""
    source = _strip_background_method_sections(_strip_reference_sections(str(text or "")))
    named_experiments = _extract_named_experiments(source)
    environment_source = _sentences_with_term_actions(
        source,
        _ENVIRONMENT_TERMS,
    )
    dataset_source = _sentences_with_term_actions(
        source,
        _DATASET_TERMS,
    )
    environments = _extract_term_group(environment_source or source, _ENVIRONMENT_TERMS, context_words=_ENVIRONMENT_CONTEXT_WORDS)
    datasets = _extract_term_group(dataset_source or source, _DATASET_TERMS, context_words=_DATASET_CONTEXT_WORDS)
    methods = _extract_method_group(source)
    metrics = _extract_term_group(source, _METRIC_TERMS, context_words=_METRIC_CONTEXT_WORDS)
    artifacts = _extract_result_artifacts(source)
    parameter_sweeps = _extract_parameter_sweeps(source)
    trend_obligations = _extract_trend_obligations(source)
    protocol_obligations = _extract_protocol_obligations(source)
    fixed_hyperparameters = _extract_fixed_hyperparameters(source)
    implementation_obligations = _extract_implementation_obligations(
        source,
        datasets=datasets,
        environments=environments,
        methods=methods,
        metrics=metrics,
        artifacts=artifacts,
        parameter_sweeps=parameter_sweeps,
        fixed_hyperparameters=fixed_hyperparameters,
        protocol_obligations=protocol_obligations,
    )
    requires_matrix = bool(
        named_experiments
        or parameter_sweeps
        or trend_obligations
        or protocol_obligations
        or fixed_hyperparameters
        or implementation_obligations
        or (len(environments) >= 2 and bool(methods))
        or (len(datasets) >= 2 and bool(methods))
        or (len(methods) >= 2 and any(word in source.lower() for word in ("experiment", "evaluation", "compare", "comparison")))
    )
    return {
        "requires_evidence_matrix": requires_matrix,
        "named_experiments": named_experiments,
        "environments": environments,
        "datasets": datasets,
        "methods": methods,
        "metrics": metrics,
        "artifacts": artifacts,
        "parameter_sweeps": parameter_sweeps,
        "trend_obligations": trend_obligations,
        "protocol_obligations": protocol_obligations,
        "fixed_hyperparameters": fixed_hyperparameters,
        "implementation_obligations": implementation_obligations,
    }


def _missing_items(required: list[dict[str, Any]], candidate_text: str, *, require_values: bool = False) -> list[str]:
    missing: list[str] = []
    for item in required:
        name = str(item.get("name", "") or "").strip()
        aliases = list(item.get("aliases", []) or [item.get("name", "")])
        aliases = list(_dedupe([*aliases, *_ABSTRACT_TERM_COVERAGE_ALIASES.get(name, ())]))
        if not _has_any_alias(candidate_text, aliases):
            missing.append(name or str(aliases[0]))
            continue
        if require_values:
            values = [str(value) for value in list(item.get("values", []) or []) if str(value).strip()]
            meaningful_values = [value for value in values if value not in {"0", "1"}]
            required_values = meaningful_values if meaningful_values else values
            if len(required_values) >= 2 and not all(_contains_alias(candidate_text, value) for value in required_values[:6]):
                missing.append(name or str(aliases[0]))
    return missing


def _trend_missing(required: list[dict[str, Any]], candidate_text: str) -> list[str]:
    lowered = str(candidate_text or "").lower()
    normalized = _normalized_text(lowered)
    missing: list[str] = []
    for item in required:
        name = str(item.get("name", "") or "")
        if name == "endpoint_low":
            has_p0 = any(token in lowered for token in ("p=0", "p = 0", "p==0", "p == 0")) or "0.0" in lowered
            has_p1 = any(token in lowered for token in ("p=1", "p = 1", "p==1", "p == 1")) or "1.0" in lowered
            has_low = any(word in normalized for word in ("endpoint", "boundary", "lowest", "minimum", "worst", "low at endpoints"))
            if not (has_p0 and has_p1 and has_low):
                missing.append(name)
        elif name == "sweep_insensitive":
            if not any(word in normalized for word in ("not sensitive", "insensitive", "stable", "robust")):
                missing.append(name)
        elif name == "baseline_outperformance":
            if not any(word in normalized for word in ("outperform", "better than", "improve", "comparison")):
                missing.append(name)
        elif not _has_any_alias(candidate_text, list(item.get("aliases", []) or [name])):
            missing.append(name)
    return missing


def evidence_contract_gaps(required_contract: dict[str, Any], candidate_text: str) -> dict[str, list[str]]:
    """Return missing evidence-matrix terms from candidate text."""
    if not required_contract.get("requires_evidence_matrix"):
        return {}
    gaps = {
        "named_experiments": _missing_items(list(required_contract.get("named_experiments", []) or []), candidate_text),
        "environments": _missing_items(list(required_contract.get("environments", []) or []), candidate_text),
        "datasets": _missing_items(list(required_contract.get("datasets", []) or []), candidate_text),
        "methods": _missing_items(list(required_contract.get("methods", []) or []), candidate_text),
        "metrics": _missing_items(list(required_contract.get("metrics", []) or []), candidate_text),
        "artifacts": _missing_items(list(required_contract.get("artifacts", []) or []), candidate_text),
        "parameter_sweeps": _missing_items(
            list(required_contract.get("parameter_sweeps", []) or []),
            candidate_text,
            require_values=True,
        ),
        "trend_obligations": _trend_missing(list(required_contract.get("trend_obligations", []) or []), candidate_text),
        "protocol_obligations": _missing_items(list(required_contract.get("protocol_obligations", []) or []), candidate_text),
        "fixed_hyperparameters": _missing_items(list(required_contract.get("fixed_hyperparameters", []) or []), candidate_text),
    }
    return {key: values for key, values in gaps.items() if values}


def implementation_obligation_gaps(required_contract: dict[str, Any], candidate_text: str) -> dict[str, list[str]]:
    """Return missing executable code/config surfaces from a paper evidence contract."""
    obligations = list(required_contract.get("implementation_obligations", []) or [])
    missing = _missing_items(obligations, candidate_text)
    return {"implementation_obligations": missing} if missing else {}


def evidence_contract_passed(required_contract: dict[str, Any], candidate_text: str) -> bool:
    return not evidence_contract_gaps(required_contract, candidate_text)


def flatten_evidence_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Return a small JSON-friendly view for review artifacts."""
    return {
        "requires_evidence_matrix": bool(contract.get("requires_evidence_matrix")),
        "named_experiments": [item.get("name", "") for item in list(contract.get("named_experiments", []) or [])],
        "environments": [item.get("name", "") for item in list(contract.get("environments", []) or [])],
        "datasets": [item.get("name", "") for item in list(contract.get("datasets", []) or [])],
        "methods": [item.get("name", "") for item in list(contract.get("methods", []) or [])],
        "metrics": [item.get("name", "") for item in list(contract.get("metrics", []) or [])],
        "artifacts": [item.get("name", "") for item in list(contract.get("artifacts", []) or [])],
        "parameter_sweeps": [
            {"name": item.get("name", ""), "values": list(item.get("values", []) or [])}
            for item in list(contract.get("parameter_sweeps", []) or [])
        ],
        "trend_obligations": [item.get("name", "") for item in list(contract.get("trend_obligations", []) or [])],
        "protocol_obligations": [item.get("name", "") for item in list(contract.get("protocol_obligations", []) or [])],
        "fixed_hyperparameters": [item.get("name", "") for item in list(contract.get("fixed_hyperparameters", []) or [])],
        "implementation_obligations": [item.get("name", "") for item in list(contract.get("implementation_obligations", []) or [])],
    }


def object_values(item: Any, names: tuple[str, ...]) -> list[str]:
    """Extract string-ish fields from pydantic models, dicts, and nested inventory dicts."""
    values: list[str] = []
    for name in names:
        value = item.get(name) if isinstance(item, dict) else getattr(item, name, None)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, (list, tuple, set)):
            values.extend(str(part) for part in value if str(part).strip())
        elif isinstance(value, dict):
            for key, nested in value.items():
                values.append(str(key))
                if isinstance(nested, (list, tuple, set)):
                    values.extend(str(part) for part in nested if str(part).strip())
                elif str(nested).strip():
                    values.append(str(nested))
    return values
