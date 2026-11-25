# EchoMatch research handoff

Updated November 19, 2026. This is a summary of a research workflow discussed with ChatGPT. It describes known code behavior and experiments, but does not imply that all local files have been pushed to GitHub. Inspect Git status and the current branch before modifying anything.

## Goal and current direction

Explore whether overlap information can improve non-rigid partial-to-partial shape matching with EchoMatch. Crucially, a deployable predicted-overlap method should use the network's predicted overlap scores at test time, not GT overlap. The supervisor also raised the possibility of cutting the two overlapping submeshes and treating them as a full-to-full matching problem; this is distinct from simply zero-masking learned features.

## EchoMatch data flow (based on inspected implementation)

- EchoMatch extracts per-vertex features F_X and F_Y with a DiffusionNet and predicts overlap probabilities using correspondence reflection followed by a separate overlap DiffusionNet.
- During standard training, feed_data() with prune_strategy='f2_only' masks Y's features by GT partiality mask when computing the functional map; the predicted masks are supervised by overlap BCE but are not directly supplied to FMReg.
- During validation/test, validate_single() computes direct feature nearest-neighbor matches and an unmasked FM. An experimental oracle variant adds an additional FM with the GT Y-overlap mask.
- Tensor shapes seen: feature Y (1,M,d), mask Y (M,), so it is broadcast to (1,M,1). The distinction between FM-derived point matches and descriptor NN matches matters.
- Training losses: FM loss, InfoNCE and overlap BCE. w_acc=0 removes BCE's contribution to gradient updates without physically deleting the overlap network.
- Validation uses validate_single() inside the train.py validation loop. Studied best-checkpoint selection used direct point-matching error, not oracle-FM error; do not change casually.

## Data and split

Dataset: PARTIALSMAL. Original training has 304 directional .vts map files representing 152 unordered pairs; official test has 100 directional maps representing 50 unordered pairs. Forward and reverse map files are paired.

A reproducible pair-level split created with seed 1234 has 272 directional training maps (136 unordered pairs) and 32 directional validation maps (16 unordered pairs). Shapes may recur across the pair-level split, so this is NOT a shape-disjoint validation split. The split is stored on Marvin scratch using symlinks to original shapes and a manifest. The official test split should remain unused during training and checkpoint selection.

## Results to date

### Oracle evaluation of the published pretrained model on official test (100 maps)

- Direct feature NN / point: 0.0914 mean geodesic error.
- Original FM / fmap: 0.1057.
- Oracle FM / fmap_oracle (GT mask applied to Y features): 0.0999.

Relative reduction versus original FM is about 5.5%. This experiment does NOT establish that the actual predicted mask yields a benefit.

### Training without overlap BCE

- Initial experiments trained EchoMatch from scratch with FM + InfoNCE losses, w_acc=0; architecture retained the overlap prediction branch.
- Two-epoch pilots finished for control w_acc=1 and no-overlap w_acc=0 on the same 272 train / 32 validation maps; they were smoke tests, not definitive comparisons.
- One four-hour w_acc=0 experiment reached approximately epoch 412, iteration 111800, before SLURM TIMEOUT. Last periodic checkpoint was 111000.pth; best.pth also present in the model output directory on Marvin.
- Last observed VALIDATION numbers (not official test): point 0.0442, fmap 0.0571, fmap_oracle 0.0552.
- The matching long-run three-loss control (w_acc=1) has been planned; completion is not confirmed in this handoff.
- Do not compare those validation results to the pretrained checkpoint's official-test numbers as if the conditions were matched.

## Proposed controlled experiments

A. Train the three-loss baseline with exactly the same split, architecture, initialization seed 1234, optimizer, scheduler, validation/checkpoint policy and approximately the same optimization budget as the no-overlap run. Select with the existing point metric and evaluate on the frozen official test only afterward.

B. Without retraining, add predicted overlap as a separate FM variant in validate_single(). For example, confirm overlap_score21 represents per-vertex probabilities of Y, reshape as (1,M,1), and multiply F_Y by the soft mask before FMReg. Preserve unmasked fmap, oracle fmap_oracle and point baselines. Compare all on the same 100 test pairs, using the same weights. GT is for scoring or the explicitly labeled oracle only.

C. Explore end-to-end overlap-aware FM: during training use predicted probabilities instead of (or in a curriculum with) GT overlap to condition FMReg while retaining FM, InfoNCE and BCE losses. Before code, inspect the FM solver differentiability; check gradients flow from matching losses through the mask to overlap-predictor parameters. Consider mask collapse, early low-quality predictions, weak conditioning and differences between training and inference. User wants a theoretical discussion with ChatGPT before implementation.

D. Separately evaluate submesh extraction / recomputed Laplace-Beltrami basis on matched submeshes, which changes geometry and boundary conditions. No code yet.

## Actual infrastructure

- Fork: https://github.com/angelhiguera/DeepShapeMatchingKit
- Remote default branch is master (not main).
- On November 19, the published experiment/oracle-overlap-fmap branch had the code for oracle inference and two SLURM eval scripts; local experimental training branch/files may not be pushed.
- Important local filenames seen in Marvin workflow: scripts/create_partialsmal_split.py; options/echo_match/train/echo_match_psmal_xyz_control.yaml; echo_match_psmal_xyz_no_overlap.yaml; pilot_2epochs.yaml; train_pilot_echomatch.sbatch; train_nightly_echomatch.sbatch. These are NOT assumed present remotely.
- The repository .gitignore ignores *.yaml, so new YAML files may require explicit git add -f, or a reviewed ignore-rule change.
- Python 3.10.8 venv on Marvin; SLURM job loads Python module first. GPU jobs only on compute nodes.

## Communication preferences

Keep the research reasoning explicit, incremental, and falsifiable. Explain each edit's purpose; avoid large unexplained refactors. Codex is for code and reviewable PRs; ChatGPT is for hypothesis design, equations, related work and interpreting experiments. Confirm tracked and pushed state before claiming anything is merged or available on GitHub.

