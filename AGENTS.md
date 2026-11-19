# Repository instructions for Codex

Read docs/RESEARCH_CONTEXT.md before changing research code. The repository is a fork of xieyizheng/DeepShapeMatchingKit, used for research on EchoMatch, partial-to-partial shape matching, overlap estimation and functional maps.

## Collaboration rules

- First inspect the actual code and current branch; the research notes are a handoff, not an authoritative description of the live working tree.
- Explain the mathematical reason for changes and identify expected tensor dimensions, gradients, evaluation metrics, and a minimal test before editing. Prefer small reviewable diffs.
- Do not silently alter baselines. Keep original feature-NN ('point'), original functional-map ('fmap'), GT oracle ('fmap_oracle') and proposed predicted-overlap ('fmap_predicted') variants distinct.
- Work on experiment/* branches or a dedicated feature branch, with a PR for review. Do not merge experimental changes to master without explicit approval.
- Never use GT overlap as an input for a proposed deployable test-time method; GT masks are allowed for supervision and clearly labeled oracle experiments.
- Do not use official PARTIALSMAL test data for model selection or hyperparameter tuning. Report split, checkpoint, metric and matching direction with each experiment.
- Do not commit checkpoint weights, datasets, generated geometry caches, local scratch links, environment files or SLURM outputs.
- Avoid brittle code modifications using automatic string replacements. Show the diff, run basic static checks, and ask for approval before submitting any GPU jobs.

## Important files

- models/echo_match_model.py — feed_data() for training and validate_single() for validation/test.
- models/partial_base_model.py — validation metrics and best-model selection.
- models/base_model.py — saving and loading weights plus optimizer/scheduler state.
- datasets/shape_dataset.py — PARTIALSMALDataset and reverse .vts map dependencies.
- options/echo_match/train/, options/echo_match/test/ — YAML experiment configurations.
- train.py and test.py — entry points.

## Marvin HPC workflow

The University of Bonn Marvin cluster is accessed over SSH. Run large experiments only on SLURM-allocated compute nodes, not the login node. In SLURM jobs, load the Python module before activating the venv:

    module load Python/3.10.8-GCCcore-12.2.0
    source "$HOME/.venvs/deepshapematchingkit/bin/activate"

The relevant GPU configuration used NVIDIA A40 on mlgpu_short. All account names, data roots, checkpoints, and scratch locations are site-specific. Do not assume a cloud Codex environment has access to Marvin, datasets, checkpoints or a GPU.

## Research plan

- A: three-loss control, same 272/32 pair split and seed as the no-overlap-loss run.
- B: frozen control weights, compare default FM versus predicted-soft-overlap FM and GT oracle on identical test pairs.
- C: discuss theoretical end-to-end predicted-overlap training before implementing. Verify gradient flow through FMReg into the overlap predictor and test numerical conditioning.
- D: separately explore physically extracting shared submeshes, recomputing spectral operators, and full-to-full formulation.

When requested to implement B, first inspect validate_single() and confirm the shape/semantics of overlap_score21. Propose the minimal patch rather than modifying train logic.

