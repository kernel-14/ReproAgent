# Ablation: w/o Reference Evidence

This runner disables the reference-evidence channel while keeping ReproAgent's
implementation-requirement channel enabled.

The ablation is controlled by passing:

```bash
--no-clone-references
```

Run one PaperBench case:

```bash
python ablation/no_reference_evidence/run_ablation.py rice --stage generate
```

For a full generate+repair run:

```bash
python ablation/no_reference_evidence/run_ablation.py rice --stage repair
```
