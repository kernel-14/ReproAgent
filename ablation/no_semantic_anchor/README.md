# Ablation: w/o Semantic Anchor

This runner disables ReproAgent's paper-derived semantic anchor while keeping
reference-repository grounding enabled.

The ablation is controlled by:

```bash
export PAPERBENCH_REPRO_DISABLE_SEMANTIC_ANCHOR=1
```

Run one PaperBench case:

```bash
python ablation/no_semantic_anchor/run_ablation.py rice --stage generate
```

For a full generate+repair run:

```bash
python ablation/no_semantic_anchor/run_ablation.py rice --stage repair
```
