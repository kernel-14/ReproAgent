# Ablation: w/o Implementation Requirements

This runner disables ReproAgent's paper-derived implementation-requirement
channel while keeping reference-evidence grounding enabled.

The ablation is controlled by:

```bash
export PAPERBENCH_REPRO_DISABLE_IMPLEMENTATION_REQUIREMENTS=1
```

Run one PaperBench case:

```bash
python ablation/no_implementation_requirement/run_ablation.py rice --stage generate
```

For a full generate+repair run:

```bash
python ablation/no_implementation_requirement/run_ablation.py rice --stage repair
```
