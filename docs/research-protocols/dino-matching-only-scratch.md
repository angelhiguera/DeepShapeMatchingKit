# PARTIALSMAL DINO matching-only reference

After the advisor meeting, we want a matching-only performance reference
before improving the initial descriptors or changing the overlap predictor,
including potential VLM-based approaches. The question is how well the
current EchoMatch matching stack learns correspondence when trained from
random EchoMatch weights with its existing DINO inputs and without the
overlap prediction branch. The result applies to this architecture,
training budget, and selection protocol.

## Architecture and protocol

- Set `networks.echo_match_net.overlap_predictor: none` to omit the learned
  temperature, soft maps, cycle matrices, second DiffusionNet, and overlap
  BCE. The predictor called by the original configurations receives the two
  shape dictionaries and their normalized descriptors. Strict loading
  remaps the published checkpoint's `permutation_network.*` and
  `overlap_diffusion_net.*` keys under `overlap_predictor.`.
- Keep precomputed 768-dimensional DINO descriptors as inputs to the
  trainable EchoMatch feature extractor. The EchoMatch weights, optimizer,
  and counters start fresh.
- Keep the original 50-mode FM solver and GT Y-mask pruning
  (`prune_strategy: f2_only`) during training. The correspondence objective
  retains the FM loss (weight 1) and within-shape and cross-shape InfoNCE
  (0.1 each). Evaluate direct descriptor nearest neighbors (`point`) and
  unmasked FM (`fmap`). Use the dataset geodesics and pool errors over all
  annotated correspondences. Matching AUC integrates the direct-matching
  PCK curve on the existing 40-point grid from 0 to 0.25.
- Follow the [original DINO training configuration](../../options/echo_match/train/echo_match_psmal_dino.yaml)
  on the 304 official train maps. Use seed 1234, batch size 1, Adam at
  `1e-3` with betas `[0.9, 0.99]`, StepLR with step size 100 and
  gamma 0.5, and the 900-epoch cap. Preserve validation every 1,000
  updates and the existing early stopping. The evaluator defaults to
  `best.pth`, selected by mean geodesic error of direct point matching.
- The upstream DINO training YAML uses the 100 official test maps for
  validation and checkpoint selection. The dedicated evaluation reports
  those same maps, so treat the result as descriptive and conditional on
  that selection protocol. Compare `point` mean geodesic error and PCK
  against corresponding PARTIALSMAL DINO results in the original
  EchoMatch paper when the matching rule, normalization, and aggregation
  agree. Report `fmap` mean error separately. Mask mIoU, balanced accuracy,
  and overlap-adjusted PCK require a predictor. Training from scratch and
  removing overlap supervision are both part of this reference when
  comparing to the previously published weights.
  [EchoMatch, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_EchoMatch_Partial-to-Partial_Shape_Matching_via_Correspondence_Reflection_CVPR_2025_paper.html)

## Verification

On synthetic pairs with PyTorch 2.3.0 CPU, the default predictor matched
master `c4236ff` exactly for forward results, losses, gradients, and one
optimizer update. All six bundled EchoMatch checkpoints loaded strictly.
The matching-only train and test pipelines completed a two-update CPU smoke
run with cached synthetic descriptors, including checkpoint reload and PCK
export. The official PARTIALSMAL run on Marvin is pending.

## Commands

Run the synthetic CPU smoke test from the repository root in the existing
Python environment. It needs no PARTIALSMAL data or GPU allocation.

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -p test_echomatch_overlap_predictor.py -v
```

On Marvin, run from the repository root with the official splits available
under `../data/PARTIALSMAL`. The launcher loads the Python module, activates
the existing venv, and prints the Git revision, map-set digests, and cache
counts. It reuses the DINO and geometry caches in each split where
available. Regenerating missing DINO features requires a compatible
PyTorch3D renderer and a populated or reachable Torch Hub cache. DINOv2's
Hub revision is unpinned, so record the cache version for reproducibility.
The evaluation job starts after training completes successfully.

```bash
scratch=/lustre/scratch/data/s23ahigu_hpc-echomatch
export TORCH_HOME="$scratch/torch-cache"
train_job=$(sbatch --parsable --account=ag_ifi_blum \
  --output="$scratch/echo-dino-matching-only-train-%j.out" \
  train_partialsmal_dino_matching_only.sbatch train) &&
eval_job=$(sbatch --parsable --account=ag_ifi_blum \
  --partition=mlgpu_short --time=00:20:00 --dependency="afterok:${train_job%%;*}" \
  --output="$scratch/echo-dino-matching-only-test-%j.out" \
  train_partialsmal_dino_matching_only.sbatch evaluate) &&
printf 'Training: %s\nEvaluation: %s\n' "$train_job" "$eval_job"
```

The run directory is
`experiments/echo_match/echo_match_psmal_dino_matching_only/`, and the
default evaluation output directory is
`results/echo_match/echo_match_psmal_dino_matching_only50_/`.
The launcher verifies that the checkpoint exists and logs its SHA-256.
An optional path after `evaluate` selects an explicit checkpoint, such as
`final.pth` after normal completion, and uses a distinct result directory.
