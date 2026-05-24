# Ablation: w/o Reference Repo

This runner disables reference-repository cloning and reference-code grounding
while keeping ReproAgent's semantic anchor enabled.

The ablation is controlled by passing:

```bash
--no-clone-references
```

Run one PaperBench case:

```bash
python ablation/no_reference_repo/run_ablation.py rice --stage generate
```

For a full generate+repair run:

```bash
python ablation/no_reference_repo/run_ablation.py rice --stage repair
```
