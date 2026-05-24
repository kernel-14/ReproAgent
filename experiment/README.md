# ReproAgent Experiment Outputs

This branch stores generated PaperBench reproduction repositories from the ReproAgent experiments. The main code branch is kept separate from these generated outputs.

Directory layout:

- `experiment/gemini_full/<paper>/repo/`: full ReproAgent pipeline outputs.
- `experiment/gemini_ablation_impl/<paper>/repo/`: ablation outputs without the implementation-requirement channel.
- `experiment/gemini_ablation_ref/<paper>/repo/`: ablation outputs without the reference-evidence channel.
- `experiment/claude/<paper>/repo/`: Claude comparison outputs.

Scores and intermediate pipeline artifacts are intentionally excluded from this branch.
