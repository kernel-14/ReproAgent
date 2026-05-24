# Stay on topic with Classifier-Free Guidance

This repository contains runnable reproduction code for **"Stay on topic with Classifier-Free Guidance"**.

The default command is a bounded `runtime_smoke`: it uses deterministic toy language-model fixtures while exercising the same CFG logits, generation, metric, FLOPs, task registry, and artifact writers used by the full routes. It does not claim paper-scale benchmark scores.

## Implemented Routes

- Equation 7 CFG for autoregressive language models:
  `guided_logits = gamma * conditional_logits - (gamma - 1) * unconditional_logits`.
- Section 3.1 zero-shot benchmark registry for ARC-c, ARC-e, BoolQ, HellaSwag, PiQA, SciQ, TriviaQA, WinoGrande, and LAMBADA; GPT-2 and Pythia model routes; `gamma=1` and `gamma=1.5` comparisons.
- Section 3.2 Chain-of-Thought/self-consistency routes for GSM8K and AQuA with guidance strengths `[1, 1.1, 1.25, 1.5, 1.75, 2]`.
- Section 3.3 HumanEval/CodeGen route with pass@k aggregation and temperatures `[0.2, 0.6, 0.8]`.
- Section 4.1 FLOPs analysis based on the ELECTRA-style transformer FLOPs formula referenced in the addendum.
- Section 5 entropy/top-p/logit-rank analysis, including the "dragon over Paris" token-rank trace.
- Section 6 classifier-guidance comparison surfaces for sentiment and toxicity likelihood deltas.
- Negative-prompt CFG is implemented as a method route, though the human preference study from Section 3.4 is out of scope.

## Commands

```bash
python main.py --mode runtime_smoke --output-dir results
python scripts/run_experiments.py --mode runtime_smoke --output-dir results
python -m pytest tests
```

Full runs require installing optional dependencies and passing `--mode full`. Dataset and model acquisition are declared in `configs/experiment_matrix.yaml`; heavy assets are not downloaded during import or smoke tests.

